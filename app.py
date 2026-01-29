"""
AI PDF Q&A System - Streamlit app
Save as: app.py
Run: streamlit run app.py
"""

import os
import streamlit as st
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import tempfile
import pickle
from typing import List, Dict, Tuple, Optional

import streamlit as st
from io import BytesIO
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Optional: OpenAI generator
try:
    import openai
except Exception:
    openai = None

##########
# CONFIG
##########

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
T5_MODEL_NAME = "google/flan-t5-small"  # change to flan-t5-large if you have GPU + RAM
EMBED_DIM = 384  # dimension for all-MiniLM-L6-v2
INDEX_DIR = "faiss_index"
META_PATH = os.path.join(INDEX_DIR, "metadata.pkl")
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")

# chunking parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# number of retrieved chunks
TOP_K = 3


# ================= TASK-2 CHAT MEMORY =================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# =====================================================

# ================= BAD ANSWER DETECTOR =================
BAD_ANSWERS = [
    "not available in the document",
    "not available",
    "cannot find",
    "no information"
]

def is_bad_answer(answer: str) -> bool:
    if not answer:
        return True
    answer = answer.lower().strip()
    return any(bad in answer for bad in BAD_ANSWERS) or len(answer) < 40
# ======================================================

##########
# UTILITIES
##########

@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str = EMBED_MODEL_NAME):
    return SentenceTransformer(model_name)

@st.cache_resource(show_spinner=False)
def load_t5_model(model_name: str = T5_MODEL_NAME, device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    return tokenizer, model, device

def pdf_to_pages(pdf_bytes: bytes) -> List[Dict]:
    """Return list of dicts: {'page': int, 'text': str} with proper BytesIO fix."""
    try:
        pdf_stream = BytesIO(pdf_bytes)     
        reader = PdfReader(pdf_stream)      
    except Exception as e:
        raise ValueError(f"PDF read error: {e}")

    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append({"page": i + 1, "text": text})
    return pages

def chunk_pages(pages: List[Dict], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP) -> List[Dict]:
    """
    Turn pages into overlapping chunks. Each chunk keeps reference to page(s).
    Returns list of {'chunk_id', 'text', 'source_pages'}.
    """
    chunks = []
    chunk_id = 0
    for p in pages:
        text = p["text"].strip()
        if not text:
            continue
        start = 0
        length = len(text)
        while start < length:
            end = min(start + chunk_size, length)
            piece = text[start:end].strip()
            if piece:
                chunks.append({
                    "chunk_id": f"{p['page']}_{chunk_id}",
                    "text": piece,
                    "source_pages": [p["page"]]
                })
                chunk_id += 1
            if end == length:
                break
            start = end - chunk_overlap
    return chunks

def build_faiss_index(embeddings: np.ndarray, dim: int = EMBED_DIM) -> faiss.IndexFlatIP:
    """
    Create FAISS index (inner product) and normalize vectors for cosine similarity.
    """
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

def save_index(index: faiss.IndexFlatIP, path: str = INDEX_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)

def load_index(path: str = INDEX_PATH) -> Optional[faiss.IndexFlatIP]:
    if os.path.exists(path):
        return faiss.read_index(path)
    return None

def save_metadata(metadata: List[Dict], path: str = META_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(metadata, f)

def load_metadata(path: str = META_PATH) -> Optional[List[Dict]]:
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    # normalize for cosine when using IndexFlatIP
    faiss.normalize_L2(emb)
    return emb

def search_index(index: faiss.IndexFlatIP, query_emb: np.ndarray, k: int = TOP_K) -> Tuple[np.ndarray, np.ndarray]:
    # query_emb expected to be normalized
    D, I = index.search(query_emb, k)
    return D, I

def prepare_prompt(context_snippets: List[str], question: str) -> str:
    context = "\n\n".join(context_snippets)
    prompt = f"""
You are an AI assistant answering strictly from the provided context.

Rules:
- Do NOT answer in a single phrase.
- Answer in 3–5 complete sentences.
- Use technical terminology from the context.
- If the answer is not present, say:
  "The answer is not available in the document."

### CONTEXT:
{context}

### QUESTION:
{question}

### ANSWER:
Explain clearly and completely.
"""
    return prompt


def generate_with_t5(tokenizer, model, device, prompt: str, max_length: int = 512) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=4,
        do_sample=False,       
        early_stopping=True
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


def generate_with_openai(api_key: str, prompt: str, model: str = "gpt-4o-mini", max_tokens: int = 300) -> str:
    if openai is None:
        raise ValueError("openai package not installed")
    openai.api_key = api_key
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return resp["choices"][0]["message"]["content"].strip()

##########
# APP UI + LOGIC
##########

st.set_page_config(page_title="AI PDF Q&A (LangChain-style RAG)", layout="wide")
st.title("AI PDF Question Answering — RAG pipeline (PDF → FAISS → LLM)")

with st.sidebar:
    st.markdown("## Configuration")
    st.write("Chunk size and overlap affect retrieval quality.")
    chunk_size = st.number_input("Chunk size", value=CHUNK_SIZE, step=100)
    chunk_overlap = st.number_input("Chunk overlap", value=CHUNK_OVERLAP, step=50)
    top_k = st.number_input("Top-K retrieved chunks", value=TOP_K, step=1, min_value=1)
    use_openai = st.checkbox("Use OpenAI for generation (if unchecked, uses local FLAN-T5)", value=False)
    if use_openai:
        openai_key = st.text_input("OpenAI API Key (optional)", type="password")
    st.write("---")
    if st.button("Clear saved index"):
        if os.path.exists(INDEX_DIR):
            import shutil
            shutil.rmtree(INDEX_DIR)
            st.success("Saved index cleared.")
        else:
            st.info("No saved index found.")

uploader = st.file_uploader("Upload a PDF file", type=["pdf"], accept_multiple_files=False)

# lazy load models
with st.spinner("Loading embedding model..."):
    embed_model = load_embedding_model(EMBED_MODEL_NAME)

t5_tokenizer = t5_model = t5_device = None
if not use_openai:
    with st.spinner("Loading generator model (T5)..."):
        t5_tokenizer, t5_model, t5_device = load_t5_model(T5_MODEL_NAME)

# If an index already exists on disk, offer to load
existing_meta = load_metadata()
existing_index = load_index()

if uploader:
    # read pdf bytes
    pdf_bytes = uploader.getvalue()
    st.success(f"Received {uploader.name} ({len(pdf_bytes)/1024:.1f} KB)")

    with st.spinner("Extracting text from PDF..."):
        try:
            pages = pdf_to_pages(pdf_bytes)
        except Exception as e:
            st.error(f"Failed to parse PDF: {e}")
            st.stop()

    st.write(f"Pages extracted: {len(pages)}")
    if st.checkbox("Show first 3 pages' text (debug)"):
        for p in pages[:3]:
            st.write(f"--- Page {p['page']} ---")
            st.write(p['text'][:1000] + ("..." if len(p['text']) > 1000 else ""))

    # step 2: chunking
    with st.spinner("Chunking pages..."):
        chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    st.write(f"Chunks created: {len(chunks)}")

    # step 3: build embeddings & FAISS
    build_index = st.button("Create embeddings & build FAISS index")
    if build_index or (existing_meta is None or existing_index is None):
        with st.spinner("Computing embeddings (this can take a while for large docs)..."):
            texts = [c["text"] for c in chunks]
            embeddings = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
            # normalize for cosine
            faiss.normalize_L2(embeddings)

        st.write("Building FAISS index in memory...")
        index = build_faiss_index(embeddings, dim=embeddings.shape[1])

        st.write("Saving index and metadata to disk...")
        save_index(index, INDEX_PATH)
        save_metadata(chunks, META_PATH)
        st.success("Index built and saved locally.")
        existing_meta = chunks
        existing_index = index
    else:
        st.info("Using saved index + metadata on disk. (You can rebuild using button above.)")

    # interactive Q&A
    st.markdown("---")
    st.header("Ask questions from the uploaded PDF")
    question = st.text_input("Type your question here and press Enter")

    if question:
        # ================= STEP-2: Save user question =================
        st.session_state.chat_history.append({"role": "user", "content": question})
        # =============================================================

        st.write("Retrieving relevant passages...")
        # ensure index loaded
        index = existing_index or load_index()
        metadata = existing_meta or load_metadata()

        if index is None or metadata is None:
            st.error("Index not found — please (re)build the index.")
        else:
            # embed the question
            expanded = question + " Explain in detail with technical terms."
            q_emb = embed_model.encode([expanded], convert_to_numpy=True)
            faiss.normalize_L2(q_emb)
            D, I = index.search(q_emb, top_k)
            retrieved = []
            SIM_THRESHOLD = 0.35
            for score, idx in zip(D[0], I[0]):
                
                if score < SIM_THRESHOLD:
                    continue
                meta = metadata[int(idx)]
                retrieved.append({"score": float(score), "text": meta["text"], "pages": meta["source_pages"]})

            # ===== TASK-2 GUARDRAIL: stop if no strong matches =====
            if len(retrieved) == 0:
                st.warning("No sufficiently relevant context found. Please rephrase your question.")
                st.stop()

            # show retrieved snippets and pages
            st.subheader("Top retrieved snippets (with page numbers)")
            for i, r in enumerate(retrieved):
                st.markdown(f"**Snippet {i+1}** — score: {r['score']:.4f} — pages: {r['pages']}")
                st.write(r["text"][:800] + ("..." if len(r["text"]) > 800 else ""))

            context_snips = [r["text"] + f"\n\n(page: {r['pages']})" for r in retrieved]

            # ================= STEP-3: Inject chat memory =================
            history_text = ""
            for h in st.session_state.chat_history[-4:]:
                history_text += f"{h['role']}: {h['content']}\n"

            prompt = prepare_prompt(context_snips, history_text + "\nCurrent question: " + question)
            # =============================================================

            st.markdown("### Generating answer...")
            if use_openai and openai_key:
                with st.spinner("Generating with OpenAI..."):
                    try:
                        answer = generate_with_openai(openai_key, prompt)
                    except Exception as e:
                        st.error(f"OpenAI generation failed: {e}")
                        answer = ""
            else:
                with st.spinner(f"Generating with {T5_MODEL_NAME}..."):
                    try:
                        answer = generate_with_t5(t5_tokenizer, t5_model, t5_device, prompt, max_length=512)
                    except Exception as e:
                        st.error(f"Local generation failed: {e}")
                        answer = ""
            if answer:
                # ================= STEP-4: Save assistant answer =================
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                # ===============================================================

                # -------- Pick BEST snippet (highest similarity) ----------
                best = max(retrieved, key=lambda x: x["score"])
                best_index = retrieved.index(best) + 1
                # ----------------------------------------------------------

                st.subheader("Answer")
                st.write(answer)

                st.success(
                    f"✅ Most reliable source: Snippet {best_index} | Page {best['pages']} | Score {round(best['score'],3)}")

                st.markdown("### Supporting snippets:")
                for i, r in enumerate(retrieved):
                    tag = "⭐ Primary" if r == best else ""
                    st.write(
                        f"- Snippet {i+1} | Page {r['pages']} | Score {round(r['score'],3)} {tag}"
                    )
                # ================= TASK-3 FEEDBACK LOOP =================
                if "feedback" not in st.session_state:
                    st.session_state.feedback = []

                col1, col2 = st.columns(2)

                with col1:
                    if st.button("👍 Helpful"):
                        st.session_state.feedback.append({
                            "question": question,
                            "answer": answer,
                            "rating": "positive",
                            "best_snippet": best_index
                        })
                        st.success("Thanks! Feedback saved.")

                with col2:
                    if st.button("👎 Not Helpful"):
                        st.session_state.feedback.append({
                            "question": question,
                            "answer": answer,
                            "rating": "negative",
                            "best_snippet": best_index
                        })
                        st.warning("Feedback recorded.")

                # ========================================================
            else:
                st.info("No answer generated.")
    # summarization option
    if st.button("Generate concise summary of the document (3-6 lines)"):
        # we will create a summary by concatenating top N chunks and asking the model to summarize
        index = existing_index or load_index()
        metadata = existing_meta or load_metadata()
        if index is None or metadata is None:
            st.error("Index not found — build index first.")
        else:
            # pick top-k global (use first 8 chunks for speed)
            sample_texts = [m["text"] for m in metadata[:8]]
            summary_prompt = "Summarize the important ideas from the following excerpts in 4-6 short lines:\n\n" + "\n\n".join(sample_texts)
            if use_openai and openai_key:
                summary = generate_with_openai(openai_key, summary_prompt, model="gpt-4o-mini", max_tokens=200)
            else:
                summary = generate_with_t5(t5_tokenizer, t5_model, t5_device, summary_prompt, max_length=200)
            st.markdown("### Document Summary")
            st.write(summary)
else:
    st.info("Upload a PDF from the left to start. You can also rebuild/clear a saved index in the sidebar.")
