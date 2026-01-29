import os
import streamlit as st
from io import BytesIO
from pypdf import PdfReader
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="📄 AI PDF Assistant",
    page_icon="🤖",
    layout="centered"
)

# -------------------- HEADER --------------------
st.markdown("""
<style>
.main-title {
    font-size: 36px;
    font-weight: bold;
    text-align: center;
}
.sub-title {
    text-align: center;
    color: gray;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: #f9f9f9;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🤖 AI PDF Question Answering</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload a PDF and ask questions in simple language</div>', unsafe_allow_html=True)

# -------------------- LOAD MODELS --------------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    llm = T5ForConditionalGeneration.from_pretrained("t5-small")
    return embedder, tokenizer, llm

embedder, tokenizer, llm = load_models()

# -------------------- PDF PROCESSING --------------------
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def split_text(text, chunk_size=400):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

# -------------------- UI CARD --------------------
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📂 Upload PDF file", type=["pdf"])

    if uploaded_file:
        with st.spinner("🔍 Reading and indexing PDF..."):
            text = extract_text_from_pdf(uploaded_file)
            chunks = split_text(text)

            embeddings = embedder.encode(chunks)
            dimension = embeddings.shape[1]

            index = faiss.IndexFlatL2(dimension)
            index.add(np.array(embeddings))

        st.success("✅ PDF processed successfully!")

        question = st.text_input("❓ Ask a question from the PDF")

        if st.button("🚀 Get Answer") and question:
            with st.spinner("🧠 Thinking..."):
                q_embedding = embedder.encode([question])
                _, indices = index.search(np.array(q_embedding), k=3)

                context = " ".join([chunks[i] for i in indices[0]])

                input_text = f"question: {question} context: {context}"
                input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

                outputs = llm.generate(input_ids, max_length=120)
                answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

            st.markdown("### ✅ Answer")
            st.info(answer)

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- FOOTER --------------------
st.markdown("""
<hr>
<center>
Made with ❤️ using Streamlit & Transformers
</center>
""", unsafe_allow_html=True)
