# STEP 4: STORE DOCUMENTS + EMBEDDINGS IN CHROMADB
# This is the ingestion stage.
# Run this when creating/rebuilding the database.

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")


# --------------------------------------------------
# 2. Load the PDF
# --------------------------------------------------

loader = PyPDFLoader("document_loaders/compliscan_report_3.pdf")

docs = loader.load()

print("Pages loaded:", len(docs))


# --------------------------------------------------
# 3. Split PDF into smaller chunks
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(docs)

print("Chunks created:", len(chunks))


# --------------------------------------------------
# 4. Create Google embedding model
# --------------------------------------------------

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=api_key
)

# --------------------------------------------------
# 5. Store chunks + embeddings in ChromaDB
# --------------------------------------------------

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)


# --------------------------------------------------
# 6. Check how many documents are stored
# --------------------------------------------------

print("Chroma database created successfully!")
print("Documents stored:", vector_db._collection.count())