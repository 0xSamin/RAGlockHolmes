from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class TextChunker:
    def __init__(self, chunk_size=400, chunk_overlap=50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk_documents(self, content):
        # 1. Convert your list of dicts to LangChain Document objects
        langchain_docs = []
        for page in content:
            doc = Document(
                page_content=page["text"],
                metadata={"page_number": page["page_number"]}
            )
            langchain_docs.append(doc)

        # 2. LangChain automatically preserves metadata for each chunk!
        # If a chunk comes from page 5, it will have {"page_number": 5}
        chunks = self.text_splitter.split_documents(langchain_docs)

        # 3. Add chunk IDs
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i

        return chunks
