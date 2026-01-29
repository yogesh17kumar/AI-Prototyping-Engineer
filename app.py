"""
AI PDF Q&A System (RAG)
Run: streamlit run app.py
"""

# ================= IMPORTS =================
import os
import pickle
from io import BytesIO
from typing import List, Dict, Tuple, Optional

import streamlit as st
import numpy as np
import faiss
import torch

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Optional OpenAI
try:
    import openai
except Exception:
    openai = None

# ================= CONFIG =================
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
T5_MODEL_NAME = "google/flan-t5-small"
EMBED_DIM = 384

INDEX_DIR = "faiss_index"
META_PATH = os.path.join(INDEX_DIR, "metadata.pkl")
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3

# ================= SESSION =================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ================= UTILITIES =================
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer(EMBED_MODEL_NAME)

@st.cache_resource(show_spinner=False)
def load_t5():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_NAME).to(device)
    return tokenizer, model, device

def pdf_to_pages(pdf_bytes: bytes) -> List[Dict]:
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []
    for i, p in enumerate(reader.pages):
        pages.append({"page": i + 1, "text": p.extract_text() or ""})
    return pages

def chunk_pages(pages, chunk_size, overlap):
    chunks = []
    cid = 0
    for p in pages:
        text = p["text"]
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append({
                "chunk_id": cid,
                "text": text[start:end],
                "page": p["page"]
            })
            cid += 1
            start = end - overlap
    return chunks

def save_index(index):
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

def load_index():
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    return None

def save_meta(meta):
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

def load_meta():
    if os.path.exists(META_PATH):
        with open(META_PATH, "rb") as f:
            return pickle.load(f)
    return None

def generate_t5(tokenizer, model, device, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(**inputs, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ================= UI =================
st.set_page_config(page_title="AI PDF Q&A", page_icon="📄", layout="wide")

st.markdown("""
<h1 style="text-align:center;">📄 AI PDF Question Answering</h1>
<p style="text-align:center;color:gray;">
PDF → Embeddings → FAISS → LLM (RAG)
</p>
<hr>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    chunk_size = st.slider("Chunk Size", 200, 1200, CHUNK_SIZE, 100)
    chunk_overlap = st.slider("Chunk Overlap", 50, 300, CHUNK_OVERLAP, 50)
    top_k = st.slider("Top-K Results", 1, 6, TOP_K)

    use_openai = st.checkbox("Use OpenAI (optional)")
    if use_openai:
        openai_key = st.text_input("OpenAI API Key", type="password")

    if st.button("🗑️ Clear Index"):
        if os.path.exists(INDEX_DIR):
            import shutil
            shutil.rmtree(INDEX_DIR)
            st.success("Index cleared")

# ================= LOAD MODELS =================
embed_model = load_embedding_model()
if not use_openai:
    t5_tokenizer, t5_model, t5_device = load_t5()

# ================= FILE UPLOAD =================
st.markdown("## 📤 Upload PDF")
pdf = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")

index = load_index()
metadata = load_meta()

if pdf:
    pdf_bytes = pdf.getvalue()
    st.success(f"Uploaded: {pdf.name}")

    with st.container(border=True):
        pages = pdf_to_pages(pdf_bytes)
        st.write(f"Pages extracted: {len(pages)}")

        chunks = chunk_pages(pages, chunk_size, chunk_overlap)
        st.write(f"Chunks created: {len(chunks)}")

        if st.button("🚀 Build Vector Index"):
            texts = [c["text"] for c in chunks]
            embeddings = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
            faiss.normalize_L2(embeddings)

            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)

            save_index(index)
            save_meta(chunks)

            st.success("Index built successfully")

    # ================= Q&A =================
    st.markdown("## 💬 Ask a Question")
    question = st.text_input("Type your question")

    if question and index:
        q_emb = embed_model.encode([question], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)

        D, I = index.search(q_emb, top_k)

        retrieved = []
        for score, idx in zip(D[0], I[0]):
            if score > 0.35:
                c = metadata[idx]
                retrieved.append(c["text"])

        with st.expander("📌 Retrieved Context"):
            for r in retrieved:
                st.write(r[:700] + "...")

        prompt = f"""
Answer the question using ONLY the context below.
Answer in 3-5 sentences.

Context:
{retrieved}

Question:
{question}
"""

        if use_openai and openai_key:
            openai.api_key = openai_key
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            answer = resp["choices"][0]["message"]["content"]
        else:
            answer = generate_t5(t5_tokenizer, t5_model, t5_device, prompt)

        with st.container(border=True):
            st.markdown("### ✅ Answer")
            st.write(answer)

else:
    st.info("👈 Upload a PDF to start")
