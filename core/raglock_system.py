from loaders.pdf_loader import PdfLoader
from chunking.text_chunker import TextChunker
from vectorstore.vectordb_manager import VectorDBManager
from utils.verifier import ResponseVerifier
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate


class RaglockSystem:
    def __init__(self, model_name="BAAI/bge-base-en-v1.5", device="cpu", llm_model="llama3.2"):
        self.chunker = TextChunker()
        self.vector_db = VectorDBManager(model_name=model_name, device=device)
        self.verifier = ResponseVerifier()

        self.llm = ChatOllama(model=llm_model, temperature=0.1)
        self.retriever = None

        self.qa_prompt = PromptTemplate.from_template("""
        You are an academic assistant. Answer the question based ONLY on the provided context.
        If you don't know the answer, say "I don't know".
        For every claim, cite the source using [Page X].

        Context:
        {context}

        Question: {question}

        Answer:
        """)

    def ingest_document(self, pdf_path):
        # Instantiate loader with path, then read
        loader = PdfLoader(pdf_path)
        raw_docs = loader.read()

        # Call the specific method
        chunked_docs = self.chunker.chunk_documents(raw_docs)

        self.vector_db.build_database(chunked_docs)
        print(f"Success: {pdf_path} ingested and stored in database.")

    def ask_question(self, query):
        """بازیابی اسناد و تولید جواب توسط LLM"""
        print(f"Searching for: {query}...")

        retriever = self.vector_db.get_retriever(search_kwargs={"k": 4}, weights=[0.4, 0.6])

        docs = retriever.invoke(query)

        if not docs:
            return "No relevant information found in the documents.", []

        context_parts = []
        for doc in docs:
            page_num = doc.metadata.get('page', 'Unknown')
            context_parts.append(f"[Page {page_num}]: {doc.page_content}")

        context_str = "\n\n".join(context_parts)

        chain = self.qa_prompt | self.llm
        response = chain.invoke({"context": context_str, "question": query})

        return response.content, docs
