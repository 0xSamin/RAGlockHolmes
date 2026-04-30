import streamlit as st
import tempfile
import os

from core.raglock_system import RaglockSystem


st.set_page_config(page_title="RAGlock Holmes", page_icon="🕵️‍♂️", layout="wide")


if "rag_system" not in st.session_state:
    st.session_state.rag_system = RaglockSystem()

    st.session_state.doc_processed = False

st.title("🕵️‍♂️ RAGlock Holmes: Academic Research Assistant")

with st.sidebar:
    st.header("Document Management")
    uploaded_file = st.file_uploader("Upload your scientific PDF/Word", type=["pdf"])

    if uploaded_file is not None and not st.session_state.doc_processed:
        with st.spinner("Processing document... (Extracting, chunking, Embedding)"):

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            st.session_state.rag_system.ingest_document(tmp_path)

            st.session_state.doc_processed = True
            st.success("Document successfully processed and indexed!")

            os.remove(tmp_path)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

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

                sources = set()
                for doc in source_docs:
                    page_num = doc.metadata.get('page', 'Unknown')
                    sources.add(f"Page {page_num}")

                source_str = ", ".join(sources) if sources else "Unknown Source"

                response_text = f"{answer_text}\n\n**Sources:** *{source_str}*"
                st.markdown(response_text)

    st.session_state.messages.append({"role": "assistant", "content": response_text})
