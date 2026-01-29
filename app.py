"""
AI PDF Q&A System (RAG)
Run: streamlit run app.py
"""

# ================= IMPORTS =================
import os
import pickle
from io import BytesIO
from typing import List, Dict

import streamlit as st
import numpy as np
import faiss
import torch

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ================= CONFIG =================
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-small"

INDEX_DIR = "faiss_index"
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
META_PATH = os.path.join(INDEX_DIR, "meta.pkl")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3
SIM_THRESHOLD = 0.35

# ================= SESSION =================
if "chat" not in st.session_state:
    st.session_state.chat = []

# ================= MODELS =================
@st.cache_resource(show_spinner=False)
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL)

@st.cache_resource(show_spinner=False)
def load_generator():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = T5Tokenizer.from_pretrained(GEN_MODEL)
    mod = T5ForConditionalGeneration.from_pretrained(GEN_MODEL).to(device)
    return tok, mod, device

# ================= PDF SAFE EXTRACT =================
def pdf_to_pages(pdf_bytes: bytes) -> List[Dict]:
    pages = []
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
    except Exception as e:
        st.error(f"PDF load failed: {e}")
        return pages

    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
            if text is None:
                text = ""
        except Exception:
            text = ""
        pages.append({"page": i + 1, "text": text})
    return pages

# ================= SAFE CHUNKING =================
def chunk_pages(pages, chunk_size, overlap):
    if overlap >= chunk_size:
        overlap = chunk_size // 2  # 🔥 HARD FIX

    chunks = []
    cid = 0

    for p in pages:
        text = p["text"]
        if not text.strip():
            continue

        start = 0
        length = len(text)

        while start < length:
            end = min(start + chunk_size, length)
            chunks.append({
                "id": cid,
                "text": text[start:end],
                "page": p["page"]
            })
            cid += 1

            if end == length:
                break
            start = end - overlap

    return chunks

# ================= INDEX =================
def save_index(index, meta):
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

def load_index():
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)
        return index, meta
    return None, None

# ================= GENERATION =================
def generate_answer(tok, model, device, prompt):
    inputs = tok(prompt, return_tensors="pt", truncation=True).to(device)
    out = model.generate(**inputs, max_length=512)
    return tok.decode(out[0], skip_special_tokens=True)

# ================= UI =================
st.set_page_config("AI PDF Q&A", "📄", layout="wide")

st.markdown("""
<h1 style="text-align:center;">📄 AI PDF Question Answering</h1>
<p style="text-align:center;color:gray;">
Stable RAG Pipeline — PDF → FAISS → LLM
</p><hr>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    chunk_size = st.slider("Chunk size", 300, 1000, CHUNK_SIZE, 100)
    chunk_overlap = st.slider("Chunk overlap", 50, 300, CHUNK_OVERLAP, 50)
    top_k = st.slider("Top-K results", 1, 6, TOP_K)

    if st.button("🗑️ Clear Index"):
        if os.path.exists(INDEX_DIR):
            import shutil
            shutil.rmtree(INDEX_DIR)
            st.success("Index cleared")

# ================= LOAD MODELS =================
embed_model = load_embed_model()
tok, gen_model, device = load_generator()

# ================= FILE UPLOAD =================
st.markdown("## 📤 Upload PDF")
pdf = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")

index, metadata = load_index()

if pdf:
    pdf_bytes = pdf.getvalue()
    st.success(f"Uploaded: {pdf.name}")

    with st.container(border=True):
        st.markdown("### 📑 Processing Document")

        pages = pdf_to_pages(pdf_bytes)
        st.write("Pages extracted:", len(pages))

        if len(pages) == 0:
            st.error("No text found. This PDF may be scanned.")
            st.stop()

        chunks = chunk_pages(pages, chunk_size, chunk_overlap)
        st.write("Chunks created:", len(chunks))

        if st.button("🚀 Build Vector Index"):
            texts = [c["text"] for c in chunks]
            emb = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
            faiss.normalize_L2(emb)

            index = faiss.IndexFlatIP(emb.shape[1])
            index.add(emb)

            save_index(index, chunks)
            metadata = chunks
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
            if score >= SIM_THRESHOLD:
                retrieved.append(metadata[idx])

        if not retrieved:
            st.warning("No relevant content found. Try rephrasing.")
            st.stop()

        with st.expander("📌 Retrieved Context"):
            for r in retrieved:
                st.markdown(f"**Page {r['page']}**")
                st.write(r["text"][:700] + "...")

        context = "\n\n".join(
            [f"(Page {r['page']}) {r['text']}" for r in retrieved]
        )

        prompt = f"""
Answer strictly from the context.
Answer in 3–5 sentences.

Context:
{context}

Question:
{question}
"""

        with st.spinner("✍️ Generating answer..."):
            answer = generate_answer(tok, gen_model, device, prompt)

        with st.container(border=True):
            st.markdown("### ✅ Answer")
            st.write(answer)

else:
    st.info("👈 Upload a text-based PDF to start")
