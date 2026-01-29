"""
Smart PDF Question Answering System
Run: streamlit run app.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from io import BytesIO
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# ================= CONFIG =================
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
T5_MODEL_NAME = "google/flan-t5-small"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3
# =========================================

# ================= PAGE UI =================
st.set_page_config(
    page_title="Smart PDF Assistant",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
<style>
.main-title {font-size:38px;font-weight:800;}
.sub-title {font-size:16px;color:gray;}
.highlight {color:#4CAF50;}
.side-card {
    background:#f9f9f9;
    padding:16px;
    border-radius:14px;
    margin-bottom:12px;
    box-shadow:0 3px 8px rgba(0,0,0,0.08);
}
.side-title {font-size:18px;font-weight:700;}
.side-desc {font-size:13px;color:gray;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📄 Smart PDF Question Answering System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">RAG-based AI using <span class="highlight">FAISS + Transformers</span></div>',
    unsafe_allow_html=True
)
st.markdown("---")
# ===========================================

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("""
    <div class="side-card">
        <div class="side-title">⚙️ Document Processing</div>
        <div class="side-desc">Controls chunking & retrieval</div>
    </div>
    """, unsafe_allow_html=True)

    chunk_size = st.slider("📦 Chunk Size", 200, 1200, CHUNK_SIZE, 100)
    chunk_overlap = st.slider("🔁 Chunk Overlap", 0, 400, CHUNK_OVERLAP, 50)
    top_k = st.slider("🎯 Top-K Results", 1, 8, TOP_K)

    st.markdown("""
    <div class="side-card">
        <div class="side-title">🧠 Model Info</div>
        <div class="side-desc">Local Transformer Model</div>
    </div>
    """, unsafe_allow_html=True)

    st.write("Model:", T5_MODEL_NAME)
# ===========================================

# ================= MODELS ==================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBED_MODEL_NAME)

@st.cache_resource
def load_t5():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = T5Tokenizer.from_pretrained(T5_MODEL_NAME)
    mdl = T5ForConditionalGeneration.from_pretrained(T5_MODEL_NAME).to(device)
    return tok, mdl, device

embed_model = load_embedding_model()
t5_tokenizer, t5_model, t5_device = load_t5()
# ===========================================

# ================= HELPERS =================
def pdf_to_pages(pdf_bytes):
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []
    for i, p in enumerate(reader.pages):
        pages.append({"page": i + 1, "text": p.extract_text() or ""})
    return pages

def chunk_pages(pages):
    chunks = []
    for p in pages:
        text = p["text"]
        for i in range(0, len(text), chunk_size - chunk_overlap):
            part = text[i:i + chunk_size]
            if part.strip():
                chunks.append({
                    "text": part,
                    "page": p["page"]
                })
    return chunks
# ===========================================

# ================= MAIN APP =================
uploader = st.file_uploader("📤 Upload a PDF document", type=["pdf"])

if uploader:
    pages = pdf_to_pages(uploader.getvalue())
    chunks = chunk_pages(pages)

    texts = [c["text"] for c in chunks]
    embeddings = embed_model.encode(texts, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    st.success(f"✅ PDF processed • {len(chunks)} chunks indexed")

    st.markdown("## ❓ Ask Questions from Your Document")
    st.caption("AI retrieves relevant sections and generates an answer")

    question = st.text_input("Type your question")

    if question:
        q_emb = embed_model.encode([question], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)

        D, I = index.search(q_emb, top_k)

        retrieved = []
        for score, idx in zip(D[0], I[0]):
            retrieved.append({
                "score": float(score),
                "text": chunks[idx]["text"],
                "page": chunks[idx]["page"]
            })

        context = "\n\n".join([r["text"] for r in retrieved])

        prompt = f"""
Answer strictly using the context below.
If not found, say answer not available.

Context:
{context}

Question:
{question}

Answer in 3–5 sentences.
"""

        inputs = t5_tokenizer(prompt, return_tensors="pt").to(t5_device)
        outputs = t5_model.generate(**inputs, max_length=400)
        answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # ===== CONFIDENCE SCORE =====
        best_score = max(r["score"] for r in retrieved)
        confidence = min(1.0, best_score * 1.4)
        confidence_percent = int(confidence * 100)

        # ===== DISPLAY =====
        st.subheader("🧠 Answer")
        st.write(answer)

        st.markdown("### 📊 Answer Confidence")
        st.progress(confidence_percent)

        if confidence_percent >= 80:
            st.success(f"High confidence — {confidence_percent}%")
        elif confidence_percent >= 60:
            st.warning(f"Medium confidence — {confidence_percent}%")
        else:
            st.error(f"Low confidence — {confidence_percent}%")

        st.markdown("### 🔍 Retrieved Evidence")
        for i, r in enumerate(retrieved):
            st.write(
                f"Snippet {i+1} | Page {r['page']} | Similarity {round(r['score'],3)}"
            )
else:
    st.info("⬅ Upload a PDF file to begin")
# ===========================================
