import os
import re
from loaders.pdf_loader import PdfLoader, DocxLoader
from chunking.text_chunker import TextChunker
from vectorstore.vectordb_manager import VectorDBManager
from utils.verifier import ResponseVerifier

# CHANGED: Swapped to LangChain's Hugging Face Integration
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from google.colab import userdata

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from utils.prompts import SYSTEM_PROMPT, QUERY_PROMPT_TEMPLATE
from typing import List, Tuple


class RaglockSystem:
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str = "cpu",
        # CHANGED: Using the official repository ID for Gemma 2 9B Instruct
        llm_model: str = "google/gemma-2-9b-it"
    ):
        print("Initializing RAGlock Holmes...")
        self.chunker = TextChunker()
        self.vector_db = VectorDBManager(model_name=model_name, device=device)
        self.verifier = ResponseVerifier()
        
        # CHANGED: Securely fetch the Hugging Face token from Colab Secrets
        try:
            hf_token = userdata.get('HF_TOKEN')
        except Exception:
            raise ValueError("❌ Please set your 'HF_TOKEN' in the Colab Secrets (key icon) sidebar.")
            
        # 1. Setup the serverless endpoint
        llm_endpoint = HuggingFaceEndpoint(
            repo_id=llm_model,
            task="text-generation",
            max_new_tokens=1024,
            temperature=0.1,
            huggingfacehub_api_token=hf_token,
        )
        
        # 2. Wrap it in a Chat-compatible interface for LangChain chat messages
        self.llm = ChatHuggingFace(llm=llm_endpoint)
        
        self.qa_prompt = PromptTemplate.from_template(QUERY_PROMPT_TEMPLATE)
        print("Initialization complete.")

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_document(self, file_path: str, original_filename: str = None):
        """Detect file type, load, chunk, and index the document."""
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext == ".pdf":
            loader = PdfLoader(file_path)
        elif ext == ".docx":
            loader = DocxLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}. Only .pdf and .docx are supported.")

        raw_docs = loader.read()
        source_name = original_filename or os.path.basename(file_path)
        chunks = self.chunker.chunk_documents(raw_docs, source_file=source_name)
        self.vector_db.build_database(chunks)

        print(f"✅ Ingested {len(chunks)} chunks from '{source_name}'")

    # ------------------------------------------------------------------
    # Q&A
    # ------------------------------------------------------------------

    def ask_question(self, question: str) -> Tuple[str, List[Document], dict]:
        """
        Answer a question using the RAG pipeline.

        Returns:
            final_answer: formatted answer string with sources appended
            docs: retrieved source documents
            verification: verification summary dict from ResponseVerifier
        """
        docs = self._retrieve(question)
        context = self._build_context(docs)
        answer = self._generate_answer(question, context)
        verification = self._verify(answer, docs)
        sources = self._build_sources(docs)

        final_answer = f"{answer}\n\n{sources}"
        return final_answer, docs, verification

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _retrieve(self, question: str) -> List[Document]:
        """Retrieve relevant chunks using hybrid BM25 + vector search."""
        k = 10 if self._is_acronym_query(question) else 6
        retriever = self.vector_db.get_retriever(
            search_kwargs={"k": k},
            weights=[0.3, 0.7]
        )
        return retriever.invoke(question)

    def _build_context(self, docs: List[Document]) -> str:
        """Format retrieved documents into a context string for the LLM."""
        parts = []
        for doc in docs:
            page = doc.metadata.get("page", "?")
            if doc.metadata.get("is_synthetic", False):
                parts.append(f"[DOCUMENT METADATA]\n{doc.page_content}")
            else:
                parts.append(f"[Page {page}]\n{doc.page_content}")
        return "\n\n".join(parts)

    def _generate_answer(self, question: str, context: str) -> str:
        """Send prompt to LLM and return the cleaned answer."""
        prompt = self.qa_prompt.format(context=context, question=question)
        
        # Standardized chat format for Gemma 2
        response = self.llm.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ])
        
        # Strip any self-generated "Sources:" section the model may append
        answer = re.split(r'\n\s*Sources?:', response.content, flags=re.IGNORECASE)[0]
        return answer.strip()

    def _verify(self, answer: str, docs: List[Document]) -> dict:
        """Run hallucination and citation checks. Logs warnings to console."""
        verification = self.verifier.get_verification_summary(answer, docs)
        if not verification["is_valid"]:
            print(f"⚠️  Verification warning: {verification['reason']}")
        return verification

    def _format_citation(self, doc: Document) -> str:
        """Format a single document chunk as a citation string."""
        source = doc.metadata.get("source", "Unknown")
        if doc.metadata.get("is_synthetic", False):
            return f"[{source} - Document Info]"
        page = doc.metadata.get("page", "?")
        return f"[{source}, Page {page}]"

    def _build_sources(self, docs: List[Document]) -> str:
        """Build a formatted sources string from retrieved documents."""
        unique_files = set(doc.metadata.get("source", "Unknown") for doc in docs)

        if len(unique_files) == 1:
            filename = list(unique_files)[0]
            pages = sorted(
                set(
                    doc.metadata.get("page", "?")
                    for doc in docs
                    if not doc.metadata.get("is_synthetic", False)
                ),
                key=lambda x: int(x) if str(x).isdigit() else float("inf")
            )
            return f"Sources ({filename}): " + ", ".join(f"Page {p}" for p in pages)
        else:
            citations = sorted(set(self._format_citation(doc) for doc in docs))
            return "Sources:\n" + "\n".join(citations)

    def _is_acronym_query(self, question: str) -> bool:
        """Detect acronym-related queries."""
        patterns = [
            r"what does \w+ stand for",
            r"what is [\w-]+\??$",
            r"expand [\w-]+",
            r"full form of [\w-]+",
            r"define [\w-]+",
        ]
        has_acronym = bool(re.search(r"\b[A-Z]{2,}(?:-[A-Za-z0-9]+)*\b", question))
        has_pattern = any(re.search(p, question.lower()) for p in patterns)
        return has_acronym or has_pattern