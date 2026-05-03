import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import streamlit as st
import tempfile
import hashlib
from core.raglock_system import RaglockSystem

st.set_page_config(page_title="RAGlock Holmes", page_icon="🕵️‍♂️", layout="wide")

@st.cache_resource
def load_rag_system():
    return RaglockSystem()

def file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()

# --- Initialize session state safely ---
if "rag_system" not in st.session_state:
    st.session_state.rag_system = load_rag_system()

if "doc_processed" not in st.session_state:
    st.session_state.doc_processed = False

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🕵️‍♂️ RAGlock Holmes: Academic Research Assistant")

with st.sidebar:
    st.header("Document Management")
    uploaded_file = st.file_uploader("Upload your scientific PDF/Word", type=["pdf", "docx"])

    if uploaded_file is not None:
        data = uploaded_file.getvalue()
        h = file_hash(data)

        # Avoid re-processing same file
        if st.session_state.get("last_file_hash") != h:
            with st.spinner("Processing document... (Extracting, chunking, Embedding)"):
                file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                    tmp_file.write(data)
                    tmp_path = tmp_file.name

                st.session_state.rag_system.ingest_document(tmp_path)

                st.session_state.doc_processed = True
                st.session_state.last_file_hash = h
                st.success("Document successfully processed and indexed!")

                os.remove(tmp_path)
        else:
            st.info("This document is already indexed.")

# --- Render chat history ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat input ---
if prompt := st.chat_input("Ask a question about the uploaded document..."):

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        if not st.session_state.doc_processed:
            response_text = "Please upload a document first before asking questions! 📁"
            st.warning(response_text)
        else:
            with st.spinner("Searching for answers..."):
                answer_text, source_docs = st.session_state.rag_system.ask_question(prompt)

                sources = {f"Page {doc.metadata.get('page', 'Unknown')}" for doc in source_docs}
                source_str = ", ".join(sources) if sources else "Unknown Source"

                response_text = f"{answer_text}\n\n**Sources:** *{source_str}*"
                st.markdown(response_text)

    st.session_state.messages.append({"role": "assistant", "content": response_text})
