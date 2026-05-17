import streamlit as st
import os
from core.raglock_system import RaglockSystem

# Force Streamlit to cache the system resource once the token is unlocked
@st.cache_resource(show_spinner="Connecting to Gemma 2 Engine...")
def load_rag_system(token_input):
    return RaglockSystem(hf_token=token_input)

# 1. Configured Sidebar Authentication Control
st.sidebar.title("🔐 Access Control")
st.sidebar.markdown("---")

# User inputs token securely through a password text box field
hf_token_input = st.sidebar.text_input("Enter Hugging Face Token:", type="password", help="Paste your hf_... token here")

if not hf_token_input:
    st.title("🕵️‍♂️ RAGlock Holmes Workspace")
    st.info("🔑 Please enter your Hugging Face Token in the sidebar to authenticate and boot the RAG engine.")
    st.stop()

# 2. Lazy load the system once the token payload exists
try:
    st.session_state.rag_system = load_rag_system(hf_token_input)
    st.sidebar.success("✅ Engine Connected to Cloud!")
except Exception as e:
    st.sidebar.error(f"❌ Connection Failed: {e}")
    st.stop()

# 3. Rest of Main User Interface Layout
st.title("🕵️‍♂️ RAGlock Holmes")
st.subheader("Advanced Academic Research Assistant")

# File Uploader Matrix
uploaded_file = st.file_uploader("Upload an Academic Document (.pdf, .docx)", type=["pdf", "docx"])

if uploaded_file:
    # Set up localized workspace directories
    os.makedirs("/content/temp_files", exist_ok=True)
    temp_path = os.path.join("/content/temp_files", uploaded_file.name)
    
    # Save the file payload to disk layout safely
    if not os.path.exists(temp_path):
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.spinner("Parsing and indexing text chunks into the Hybrid Matrix..."):
            st.session_state.rag_system.ingest_document(temp_path, original_filename=uploaded_file.name)
        st.success(f"Successfully processed document: {uploaded_file.name}")

st.markdown("---")

# Q&A Segment Interface Loops
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history profile elements
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Query input mechanics
if prompt := st.chat_input("Ask a research question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching document context and verifying references..."):
            final_answer, retrieved_docs, verification = st.session_state.rag_system.ask_question(prompt)
            st.markdown(final_answer)
            
            # Show a clear UI warning if numerical discrepancies are spotted
            if not verification["is_valid"]:
                st.warning(f"⚠️ Metadata Guardrail Warning: {verification['reason']}")
                
    st.session_state.messages.append({"role": "assistant", "content": final_answer})