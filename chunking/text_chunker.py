from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class TextChunker:
    def __init__(self, chunk_size=1200, chunk_overlap=200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk_documents(self, content, source_file=None):

        chunks = []
        chunk_id = 0

        for page in content:

            # Create base document
            doc = Document(
                page_content=page["text"],
                metadata={
                    "page": page["page_number"],
                    "is_synthetic": page.get("is_synthetic", False),
                    "author": page.get("metadata", {}).get("author"),
                    "title": page.get("metadata", {}).get("title"),
                    "source": source_file or "Unknown"
                }
            )

            # ---- Preserve metadata chunk (page 0) ----
            if page["page_number"] == 0:
                doc.metadata["chunk_id"] = chunk_id
                chunks.append(doc)
                chunk_id += 1
                continue

            # ---- Split normal pages ----
            split_docs = self.text_splitter.split_documents([doc])

            for sdoc in split_docs:
                sdoc.metadata["chunk_id"] = chunk_id
                chunks.append(sdoc)
                chunk_id += 1

        return chunks
