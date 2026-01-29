import os
import streamlit as st
from io import BytesIO
from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# ================= CONFIG =================
st.set_page_config(page_title="AI PDF Q&A", layout="wide")
st.title("📄 AI PDF Question Answering System (Fast RAG)")

# ================= API KEY =================
openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key

# ================= PDF UPLOAD =================
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and openai_key:

    # ---------- READ PDF ----------
    pdf_reader = PdfReader(BytesIO(uploaded_file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    # ---------- CHUNKING ----------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)

    # ---------- EMBEDDINGS + FAISS ----------
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)

    # ---------- LLM ----------
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o-mini"
    )

    # ---------- RAG CHAIN ----------
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    st.success("PDF processed successfully! Ask questions below 👇")

    # ================= QUESTION =================
    query = st.text_input("Ask a question from the PDF")

    if query:
        with st.spinner("Thinking..."):
            result = qa_chain(query)

        st.subheader("✅ Answer")
        st.write(result["result"])

        st.subheader("📌 Source Chunks")
        for i, doc in enumerate(result["source_documents"]):
            st.markdown(f"**Chunk {i+1}:**")
            st.write(doc.page_content[:500] + "...")

else:
    st.info("Upload PDF and enter OpenAI API key to start.")
