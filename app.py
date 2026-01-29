import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from io import BytesIO
import streamlit as st
import numpy as np
import faiss
import torch
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration

# ---------------- CONFIG ----------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-small"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3

# ---------------- UI ----------------
st.set_page_config(page_title="AI PDF Chat", layout="wide")

st.markdown("""
<style>
.main {background-color:#0f172a;}
.card {
    background:#020617;
    padding:25px;
    border-radius:15px;
    box-shadow:0 0 20px rgba(0,0,0,0.5);
}
h1,h2 {color:#38bdf8;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center'>📘 AI PDF Q&A</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center'>Chat with PDFs using RAG</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- MODELS ----------------
@st.cache_resource
def load_models():
    embed = SentenceTransformer(EMBED_MODEL)
    tokenizer = T5Tokenizer.from_pretrained(LLM_MODEL)
    model = T5ForConditionalGeneration.from_pretrained(LLM_MODEL)
    return embed, tokenizer, model

embed_model, tokenizer, llm = load_models()

# ---------------- FUNCTIONS ----------------
def extract_text(pdf):
    reader = PdfReader(BytesIO(pdf))
    return [p.extract_text() or "" for p in reader.pages]

def chunk_text(pages):
    chunks = []
    for page in pages:
        start = 0
        while start < len(page):
            end = start + CHUNK_SIZE
            chunks.append(page[start:end])
            start = end - CHUNK_OVERLAP
    return chunks

def build_index(chunks):
    emb = embed_model.encode(chunks, convert_to_numpy=True)
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index

def generate_answer(context, question):
    prompt = f"""
Answer strictly from context.
If not found say: Answer not available in the document.

Context:
{context}

Question:
{question}

Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    output = llm.generate(**inputs, max_length=300)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Top-K Chunks", 1, 5, TOP_K)

# ---------------- MAIN ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

pdf = st.file_uploader("📄 Upload PDF", type=["pdf"])

if pdf:
    pages = extract_text(pdf.getvalue())
    chunks = chunk_text(pages)

    if st.button("🔍 Build Index"):
        st.session_state.index = build_index(chunks)
        st.session_state.chunks = chunks
        st.success("Index Ready!")

    question = st.text_input("💬 Ask a question")

    if question and "index" in st.session_state:
        q_emb = embed_model.encode([question], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)

        D, I = st.session_state.index.search(q_emb, top_k)
        context = "\n\n".join([st.session_state.chunks[i] for i in I[0]])

        answer = generate_answer(context, question)
        st.subheader("✅ Answer")
        st.write(answer)
else:
    st.info("Upload a PDF to start")

st.markdown("</div>", unsafe_allow_html=True)
