"""
Streamlit frontend for the PDF RAG pipeline (LangChain + Gemini + Chroma).

Combines your two scripts into one app:
  - Sidebar: upload a PDF and build/rebuild the Chroma vector database.
  - Main area: chat with the PDF using the existing vector database.

Run with:
    streamlit run app.py
"""

import os
import shutil
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_chroma import Chroma


# --------------------------------------------------
# Config
# --------------------------------------------------

PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "gemini-embedding-001"
CHAT_MODEL_NAME = "gemini-3.7-flash"

st.set_page_config(page_title="PDF Chat (Gemini + Chroma)", page_icon="📄", layout="wide")

load_dotenv()


# --------------------------------------------------
# Helpers (cached so we don't rebuild models on every rerun)
# --------------------------------------------------

def get_api_key() -> str | None:
    # Prefer a key entered in the sidebar (session), else fall back to .env
    return st.session_state.get("api_key") or os.getenv("GOOGLE_API_KEY")


@st.cache_resource(show_spinner=False)
def get_embedding_model(api_key: str):
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME, google_api_key=api_key)


@st.cache_resource(show_spinner=False)
def get_llm(api_key: str):
    return ChatGoogleGenerativeAI(model=CHAT_MODEL_NAME, google_api_key=api_key)


def load_vector_db(api_key: str):
    embedding_model = get_embedding_model(api_key)
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)


def db_exists() -> bool:
    return os.path.isdir(PERSIST_DIR) and len(os.listdir(PERSIST_DIR)) > 0


def build_vector_db(pdf_path: str, api_key: str, chunk_size: int, chunk_overlap: int, rebuild: bool):
    if rebuild and os.path.isdir(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    embedding_model = get_embedding_model(api_key)
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=PERSIST_DIR,
    )
    return len(docs), len(chunks), vector_db._collection.count()


def extract_answer_text(response) -> str:
    """Handle both plain-string and list-of-block response.content shapes."""
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def answer_question(question: str, api_key: str, k: int, fetch_k: int):
    vector_db = load_vector_db(api_key)
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k},
    )
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = f"""
You are a helpful PDF assistant.

Answer the user's question using ONLY the information
provided in the context.

If the answer is not available in the context, say:

"I could not find the answer in the PDF."

Do not make up information.

Context:
{context}

User Question:
{question}
"""
    llm = get_llm(api_key)
    response = llm.invoke(prompt)
    return extract_answer_text(response), retrieved_docs


# --------------------------------------------------
# Sidebar — API key + ingestion
# --------------------------------------------------

with st.sidebar:
    st.header("Setup")

    env_key_present = bool(os.getenv("GOOGLE_API_KEY"))
    key_input = st.text_input(
        "Google API key",
        type="password",
        placeholder="Loaded from .env" if env_key_present else "Paste your GOOGLE_API_KEY",
        help="Leave blank to use GOOGLE_API_KEY from your .env file.",
    )
    if key_input:
        st.session_state["api_key"] = key_input

    st.divider()
    st.header("1. Build the database")

    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.number_input("Chunk size", value=1000, step=100)
    with col2:
        chunk_overlap = st.number_input("Chunk overlap", value=200, step=50)

    rebuild = st.checkbox("Rebuild (wipe existing database)", value=not db_exists())

    build_clicked = st.button("Build / update database", type="primary", use_container_width=True)

    if build_clicked:
        api_key = get_api_key()
        if not api_key:
            st.error("Enter a Google API key or set GOOGLE_API_KEY in your .env file.")
        elif not uploaded_pdf:
            st.error("Upload a PDF first.")
        else:
            with st.spinner("Loading PDF, splitting into chunks, and embedding..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_pdf.getvalue())
                    tmp_path = tmp.name
                try:
                    n_pages, n_chunks, n_stored = build_vector_db(
                        tmp_path, api_key, int(chunk_size), int(chunk_overlap), rebuild
                    )
                    st.success(f"Done: {n_pages} pages -> {n_chunks} chunks -> {n_stored} stored vectors.")
                    load_vector_db.clear() if hasattr(load_vector_db, "clear") else None
                finally:
                    os.remove(tmp_path)

    st.divider()
    st.header("2. Retrieval settings")
    k = st.slider("k (chunks used per answer)", 1, 10, 4)
    fetch_k = st.slider("fetch_k (candidates for MMR)", k, 20, 10)

    st.divider()
    st.caption(f"Database folder: `{PERSIST_DIR}` — {'found' if db_exists() else 'not found yet'}")


# --------------------------------------------------
# Main — chat
# --------------------------------------------------

st.title("📄 Chat with your PDF")
st.caption(f"Embeddings: {EMBEDDING_MODEL_NAME} · Chat model: {CHAT_MODEL_NAME} · Retrieval: MMR")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources used"):
                for i, doc in enumerate(msg["sources"], 1):
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**Chunk {i} (page {page})**")
                    st.text(doc.page_content[:500])

question = st.chat_input("Ask a question about the PDF...")

if question:
    api_key = get_api_key()
    if not api_key:
        st.error("Enter a Google API key in the sidebar or set GOOGLE_API_KEY in your .env file.")
    elif not db_exists():
        st.error("No database found yet. Upload a PDF and click 'Build / update database' in the sidebar first.")
    else:
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and asking Gemini..."):
                try:
                    answer, sources = answer_question(question, api_key, k, fetch_k)
                except Exception as e:
                    answer, sources = f"Error: {e}", []
            st.markdown(answer)
            if sources:
                with st.expander("Sources used"):
                    for i, doc in enumerate(sources, 1):
                        page = doc.metadata.get("page", "?")
                        st.markdown(f"**Chunk {i} (page {page})**")
                        st.text(doc.page_content[:500])

        st.session_state["messages"].append({"role": "assistant", "content": answer, "sources": sources})