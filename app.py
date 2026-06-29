import os
import json
import faiss
import numpy as np
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from groq import Groq
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import re

load_dotenv()

app = Flask(__name__)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory vector store
vector_store = {
    "index": None,
    "chunks": [],
    "doc_name": None
}

def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def extract_text_from_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list:
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i + size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def build_index(chunks: list):
    embedder = get_embedder()
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def retrieve(query: str, k: int = 4) -> list:
    if vector_store["index"] is None:
        return []
    embedder = get_embedder()
    q_emb = embedder.encode([query]).astype("float32")
    _, indices = vector_store["index"].search(q_emb, k)
    return [vector_store["chunks"][i] for i in indices[0] if i < len(vector_store["chunks"])]

def answer_question(question: str) -> str:
    context_chunks = retrieve(question)
    if not context_chunks:
        return "Please upload a document first so I can answer questions about it."

    context = "\n\n---\n\n".join(context_chunks)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": (
                "You are a helpful assistant that answers questions strictly based on the provided document context. "
                "If the answer is not in the context, say so clearly. Be concise and accurate."
            )},
            {"role": "user", "content": f"Context from document:\n{context}\n\nQuestion: {question}"}
        ],
        max_tokens=1024
    )
    return response.choices[0].message.content

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in ['pdf', 'txt']:
        return jsonify({'error': 'Only PDF and TXT files are supported'}), 400

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    text = extract_text_from_pdf(path) if ext == 'pdf' else extract_text_from_txt(path)
    if not text.strip():
        return jsonify({'error': 'Could not extract text from the document'}), 400

    chunks = chunk_text(text)
    index = build_index(chunks)

    vector_store["index"] = index
    vector_store["chunks"] = chunks
    vector_store["doc_name"] = file.filename

    return jsonify({
        'message': f'Document processed successfully',
        'filename': file.filename,
        'chunks': len(chunks)
    })

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question', '').strip()
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    try:
        answer = answer_question(question)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def status():
    return jsonify({
        'loaded': vector_store["doc_name"] is not None,
        'doc_name': vector_store["doc_name"],
        'chunks': len(vector_store["chunks"])
    })

if __name__ == '__main__':
    app.run(debug=True, port=5002)
