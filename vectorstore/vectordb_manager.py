from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
import pickle
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from typing import List


class HybridRetriever(BaseRetriever):
    """Combines BM25 and Vector retrievers with weighted scoring."""

    bm25_retriever: BaseRetriever
    vector_retriever: BaseRetriever
    bm25_weight: float = 0.5
    vector_weight: float = 0.5
    top_k: int = 10

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Retrieve and merge documents from both retrievers."""
        # FIX: Ensure both inner retrievers consistently utilize modern LangChain execution patterns (.invoke)
        bm25_docs = self.bm25_retriever.invoke(query)
        vector_docs = self.vector_retriever.invoke(query)

        doc_scores = {}

        # Score BM25 results
        for i, doc in enumerate(bm25_docs):
            key = doc.page_content
            score = (len(bm25_docs) - i) * self.bm25_weight
            doc_scores[key] = doc_scores.get(key, 0) + score

        # Score vector results
        for i, doc in enumerate(vector_docs):
            key = doc.page_content
            score = (len(vector_docs) - i) * self.vector_weight
            doc_scores[key] = doc_scores.get(key, 0) + score

        # Merge and deduplicate
        all_docs = {doc.page_content: doc for doc in bm25_docs + vector_docs}

        # Sort by combined score
        sorted_docs = sorted(
            all_docs.items(),
            key=lambda x: doc_scores.get(x[0], 0),
            reverse=True
        )

        return [doc for _, doc in sorted_docs[:self.top_k]]

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Async version - falls back to sync for simplicity."""
        return self._get_relevant_documents(query)


class VectorDBManager:
    def __init__(self, persist_directory="./chroma_db", model_name="BAAI/bge-base-en-v1.5", device="cpu"):
        self.persist_directory = persist_directory

        # Fixed the cache path for Linux and removed local-only restriction
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'local_files_only': False},
            encode_kwargs={"normalize_embeddings": True},
            cache_folder="/content/.cache/huggingface/hub"
        )

        self.vector_store = None

    def build_database(self, chunks):
        """Build vector database from chunks and save BM25 data."""
        # Build Chroma database
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        self.vector_store.persist()

        # Save chunks for BM25
        chunks_path = os.path.join(self.persist_directory, "bm25_chunks.pkl")
        with open(chunks_path, "wb") as f:
            pickle.dump(chunks, f)

        return self.vector_store

    def get_retriever(self, search_kwargs=None, weights=None):
        """
        Returns a HybridRetriever combining BM25 and ChromaDB.

        Args:
            search_kwargs: dict with 'k' for number of results (default: {"k": 4})
            weights: [bm25_weight, chroma_weight] (default: [0.4, 0.6])
        """
        if search_kwargs is None:
            search_kwargs = {"k": 4}
        if weights is None:
            weights = [0.4, 0.6]

        k = search_kwargs.get("k", 4)

        # Load or initialize Chroma
        if self.vector_store is None:
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )

        chroma_retriever = self.vector_store.as_retriever(search_kwargs=search_kwargs)

        # Load BM25 chunks
        chunks_path = os.path.join(self.persist_directory, "bm25_chunks.pkl")

        if os.path.exists(chunks_path):
            with open(chunks_path, "rb") as f:
                saved_chunks = pickle.load(f)

            bm25_retriever = BM25Retriever.from_documents(saved_chunks)
            bm25_retriever.k = k

            hybrid_retriever = HybridRetriever(
                bm25_retriever=bm25_retriever,
                vector_retriever=chroma_retriever,
                bm25_weight=weights[0],
                vector_weight=weights[1],
                top_k=k
            )

            return hybrid_retriever

        else:
            print("Warning: BM25 chunks not found. Using vector search only.")
            return chroma_retriever