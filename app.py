"""
AI PDF Q&A System - Streamlit app
Save as: app.py
Run: streamlit run app.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pickle
from typing import List, Dict, Tuple, Optional
from io import BytesIO

import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

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
# =========================================

# ============== SESSION ===================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# =========================================

# ============== MODELS ====================
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)

@st.cache_resource
def load_t5():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = T5Tokenizer.from_pretrained(T5_MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_NAME).to(device)
    return tok, model, device
# =========================================

# ============== HELPERS ===================
def pdf_to_pages(pdf_bytes):
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []
    for i, p in enumerate(reader.pages):
        pages.append({"page": i+1, "text": p.extract_text() or ""})
    return pages

def chunk_pages(pages, size, overlap):
    chunks = []
    cid = 0
    for p in pages:
        txt = p["text"]
        i = 0
        while i < len(txt):
            piece = txt[i:i+size].strip()
            if piece:
                chunks.append({
                    "chunk_id": cid,
                    "text": piece,
                    "source_pages": [p["page"]]
                })
                cid += 1
            i += size - overlap
    return chunks

def build_index(emb):
    faiss.normalize_L2(emb)
    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb)
    return idx

def save_all(index, meta):
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

def load_all():
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        return faiss.read_index(INDEX_PATH), pickle.load(open(META_PATH, "rb"))
    return None, None

def generate_answer(tok, model, device, prompt):
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    out = model.generate(**inp, max_length=512)
    return tok.decode(out[0], skip_special_tokens=True)
# =========================================

# ================= UI =====================
st.set_page_config("AI PDF Q&A", "📄", layout="wide")

st.markdown("""
<style>
.card {
    background:#f8f9fa;
    padding:20px;
    border-radius:16px;
    box-shadow:0 4px 10px rgba(0,0,0,.08);
}
</style>
""", unsafe_allow_html=True)

st.title("📄 AI PDF Question Answering System")
st.caption("PDF → FAISS → Transformer (RAG Pipeline)")

# ============ SIDEBAR (NEW UI) ============
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    st.markdown("<div class='card'>🔍 Retrieval Settings</div>", unsafe_allow_html=True)
    chunk_size = st.slider("Chunk Size", 200, 1200, CHUNK_SIZE, 100)
    chunk_overlap = st.slider("Chunk Overlap", 0, 400, CHUNK_OVERLAP, 50)
    top_k = st.slider("Top-K Results", 1, 8, TOP_K)

    st.markdown("<div class='card'>🧠 Generator</div>", unsafe_allow_html=True)
    use_openai = st.toggle("Use OpenAI (optional)", False)
    if use_openai:
        openai_key = st.text_input("OpenAI API Key", type="password")

    st.markdown("<div class='card'>🗑 Maintenance</div>", unsafe_allow_html=True)
    if st.button("Clear FAISS Index"):
        if os.path.exists(INDEX_DIR):
            import shutil
            shutil.rmtree(INDEX_DIR)
            st.success("Index cleared")
# =========================================

embedder = load_embedder()
tok = model = device = None
if not use_openai:
    tok, model, device = load_t5()

uploaded = st.file_uploader("Upload PDF", type="pdf")

if uploaded:
    pages = pdf_to_pages(uploaded.getvalue())
    chunks = chunk_pages(pages, chunk_size, chunk_overlap)

    if st.button("Build / Rebuild Index"):
        emb = embedder.encode([c["text"] for c in chunks], convert_to_numpy=True)
        index = build_index(emb)
        save_all(index, chunks)
        st.success("Index built successfully")

    index, meta = load_all()

    st.markdown("---")
    q = st.text_input("Ask a question from the PDF")

    if q and index:
        q_emb = embedder.encode([q], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = index.search(q_emb, top_k)

        context = ""
        for i in I[0]:
            context += meta[i]["text"] + "\n"

        prompt = f"""
Answer strictly from the context.

Context:
{context}

Question:
{q}

Answer:
"""

        if use_openai and openai_key:
            openai.api_key = openai_key
            ans = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )["choices"][0]["message"]["content"]
        else:
            ans = generate_answer(tok, model, device, prompt)

        st.markdown("### ✅ Answer")
        st.success(ans)
else:
    st.info("Upload a PDF to start.")
