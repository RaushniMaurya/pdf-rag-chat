# 📄 PDF RAG Chat

PDF RAG Chat is a Retrieval-Augmented Generation app that lets you upload any PDF and ask questions about it — powered by **LangChain**, **Google Gemini**, **ChromaDB**, and a **Streamlit** frontend.

---

## ✨ Features

- 📤 Upload a PDF and build a local vector database from it
- 💬 Ask questions about the PDF in a chat interface
- 🎯 Answers are generated **only** from retrieved context — no hallucinated info
- 🔍 View the exact source chunks used for each answer
- ⚙️ Adjustable chunk size, overlap, and retrieval settings (k / fetch_k)

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Frontend | Streamlit |
| Orchestration | LangChain |
| Embeddings | Google Gemini (`gemini-embedding-001`) |
| Chat model | Google Gemini (`gemini-3.7-flash`) |
| Vector store | ChromaDB |
| PDF loading | `PyPDFLoader` |

---

## 🚀 Setup

1. Clone the repo:
    git clone https://github.com/RaushniMaurya/pdf-rag-chat.git
    cd pdf-rag-chat


2. Install dependencies:

    pip install -r requirements.txt    


3. Create a `.env` file in the project root with your Google API key:    
    GOOGLE_API_KEY=your_api_key_here


---

## ▶️ Usage

Run the Streamlit app:
  python -m streamlit run app.py

Then in the browser tab that opens:
1. In the sidebar, upload a PDF and click **Build / update database**
2. Once the database is built, ask questions about the PDF in the chat box

---

## 📁 Project Structure  

​```
RAG/
├── app.py                  # Streamlit frontend (build DB + chat)
├── create_chroma.py        # Standalone ingestion script
├── main.py                 # Standalone Q&A script
├── document_loaders/       # Source PDFs
├── requirements.txt
└── .env                    # Not committed, holds GOOGLE_API_KEY

​```




---

## 📝 Notes

- `.env` and `chroma_db/` are git-ignored — the vector database is generated locally and never committed.
- Rebuilding the database wipes and regenerates `chroma_db/` from the uploaded PDF.