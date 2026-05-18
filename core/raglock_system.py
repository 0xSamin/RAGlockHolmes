import os
import sys
import re
import torch
from loaders.pdf_loader import PdfLoader, DocxLoader
from chunking.text_chunker import TextChunker
from vectorstore.vectordb_manager import VectorDBManager
from utils.verifier import ResponseVerifier
from unsloth import FastLanguageModel  # Directly loading local GPU acceleration kernels
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from utils.prompts import SYSTEM_PROMPT, QUERY_PROMPT_TEMPLATE
from typing import List, Tuple


class RaglockSystem:
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        llm_model: str = "unsloth/gemma-2-9b-it-bnb-4bit", 
        hf_token: str = None
    ):
        print(f"Initializing RAGlock Holmes Local Engine on GPU ({device})...")
        self.chunker = TextChunker()
        self.vector_db = VectorDBManager(model_name=model_name, device=device)
        self.verifier = ResponseVerifier()
        
        # Load the accelerated model and native fast tokenizer via Unsloth patches
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=llm_model,
            max_seq_length=2048,
            load_in_4bit=True,  # Lowers VRAM footprint to comfortably fit on Tesla T4
            device_map="auto"
        )
        FastLanguageModel.for_inference(self.model)  # Optimize model layers for generation tasks
        
        self.qa_prompt = PromptTemplate.from_template(QUERY_PROMPT_TEMPLATE)
        print("Local GPU Engine Initialization complete.")

    # ------------------------------------------------------------------
    # Ingestion Layer
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
    # Q&A Processing Layout
    # ------------------------------------------------------------------

    def ask_question(self, question: str) -> Tuple[str, List[Document], dict]:
        """
        Answer a question using the hybrid RAG pipeline matrix.
        """
        docs = self._retrieve(question)
        context = self._build_context(docs)
        answer = self._generate_answer(question, context)
        verification = self._verify(answer, docs)
        sources = self._build_sources(docs)

        final_answer = f"{answer}\n\n{sources}"
        return final_answer, docs, verification

    # ------------------------------------------------------------------
    # Private Core Helpers
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
        """Generate answer using Gemma 2 with user-optimized chat structures."""
        
        # Proper chat template structure optimized for 4-bit Gemma 2 stability
        conversation = [
            {
                "role": "user", 
                "content": f"You are a precise research assistant. Answer based ONLY on the provided context.\n\nContext:\n{context}\n\nQuestion: {question}\n\nProvide a clear, direct answer in 2-3 sentences. Do not include page numbers in your answer."
            }
        ]
        
        # Apply structural chat template to string formats first
        formatted_prompt = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Safely tokenize input sequences with explicit max token guards
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        input_length = inputs.input_ids.shape[1]
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,  # Hardcoded to EOS to secure Unsloth kernel dim matrixes
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Defend pipeline against unexpected 1D generation expansions
        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(0)
        
        generated_tokens = outputs[0, input_length:]
        
        # Secure memory migration onto host space
        generated_tokens = generated_tokens.detach().cpu()
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Cleanup output formatting leaks
        answer = answer.split('\n\n')[0]  # Take the first coherent block only
        answer = re.sub(r'\[Page \d+\]', '', answer)  # Strip inline page noise
        
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