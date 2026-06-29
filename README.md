# RAG Document Chat

## Live Demo

**https://rag-document-chat-nrwb.onrender.com**

A Retrieval-Augmented Generation (RAG) application that lets you upload any **PDF or TXT** document and chat with it using **LLaMA 3.1** and **Groq**.

## How It Works

```
Upload PDF/TXT
      |
      v
Extract & Chunk Text
      |
      v
Generate Embeddings (all-MiniLM-L6-v2)
      |
      v
Store in FAISS Vector Index
      |
      v
User Asks Question
      |
      v
Retrieve Top-4 Relevant Chunks
      |
      v
LLaMA 3.1 Answers Based on Context
```

## Tech Stack

| Category | Tools |
|---|---|
| LLM | LLaMA 3.1 8B via Groq |
| Embeddings | all-MiniLM-L6-v2 (HuggingFace) |
| Vector Store | FAISS (Facebook AI) |
| PDF Parsing | PyPDF2 |
| Web Framework | Flask |
| Frontend | HTML, CSS, JavaScript |

## Project Structure

```
rag-document-chat/
├── app.py              # RAG pipeline + Flask API
├── requirements.txt    # Dependencies
├── .env                # API keys (not committed)
├── .env.example        # Template for env variables
├── .gitignore
├── uploads/            # Uploaded documents (not committed)
└── templates/
    └── index.html      # Chat UI with file upload
```

## Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/Shrutiraj26/rag-document-chat.git
cd rag-document-chat
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set up environment**
```bash
cp .env.example .env
# Add your Groq API key to .env
```

**4. Run the app**
```bash
python app.py
```

**5. Open in browser**
```
http://localhost:5002
```

## Features

- Upload PDF or TXT documents via drag-and-drop
- Automatic text extraction and chunking (500 words, 50-word overlap)
- Semantic search using FAISS vector index
- Answers grounded strictly in uploaded document context
- Clean dark-mode chat interface

## Get a Free Groq API Key

Sign up at **https://console.groq.com** — no credit card required.
