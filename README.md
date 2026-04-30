# 🕵️‍♂️ RAGlock Holmes 
**Advanced Academic Research Assistant (Zero-Hallucination RAG System)**

RAGlock Holmes is a specialized Retrieval-Augmented Generation (RAG) pipeline designed for academic researchers. It tackles the massive volume of scientific papers by providing precise, verifiable answers with strict page-level citations, ensuring Large Language Models (LLMs) do not hallucinate academic claims.

## 🎯 The Problem
Researchers spend hours extracting methodologies and mathematical formulas (e.g., $y = \alpha + \beta X$) from PDFs. Standard LLMs hallucinate answers and lack the precise referencing required to build trust in an academic environment.

## 💡 The Solution
A two-stage RAG architecture that separates answer generation from citation verification. Every claim output by RAGlock Holmes is mechanically and intelligently checked against the source document's metadata to ensure absolute accuracy.

## ✨ Key Features
- **Zero-Hallucination Q&A:** Engineered prompts and a custom `ResponseVerifier` restrict the model strictly to the retrieved context.
- **Verifiable Citations:** Automated attachment and verification of page numbers to the final answer.
- **Structural Metadata Preservation:** Advanced semantic chunking that accurately tracks page numbers and document structure.
- **100% Local Execution:** Complete privacy for unpublished research using local vector databases and local LLMs.

## 🛠️ Tech Stack & Architecture
- **Language:** Python (OOP, modular design)
- **Orchestration:** LangChain
- **Local LLM:** Llama 3.2 3B Instruct (via Ollama)
- **Embedding Model:** `BAAI/bge-base-en-v1.5` (Optimized for RAG)
- **Vector Database:** ChromaDB (Local storage)
- **Document Ingestion:** `pypdf` (PDFs) + `python-docx` (Word Documents)
- **User Interface:** Streamlit

## 🚀 Installation & Setup

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/raglock-holmes.git
cd raglock-holmes

**2. Install Python dependencies**
bash
pip install -r requirements.txt

**3. Setup Local LLM (Ollama)**
Download and install [Ollama](https://ollama.com/), then pull the Llama 3.2 model:
bash
ollama run hf.co/unsloth/Llama-3.2-3B-Instruct-GGUF:UD-Q4_K_XL

**4. Run the application**
bash
streamlit run app.py

## 🗺️ Roadmap (v2.0)
- [x] Base architecture (Loaders, Chunkers, Embeddings)
- [x] Local LLM Integration (Llama 3.2)
- [ ] Hybrid Retrieval Implementation (BM25 + Dense Vector)
- [ ] Advanced Hallucination Detection Layer (Regex/NER)
- [ ] Custom Evaluation Framework
- [ ] Deployment & Optimization

## 📄 License
MIT License

