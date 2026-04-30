from pypdf import PdfReader
from docx import Document


class BaseLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def read(self):
        raise NotImplementedError("Subclasses must implement the 'read' method.")


class PdfLoader(BaseLoader):
    def read(self):
        paper = PdfReader(self.file_path)
        text_data = []
        for i, page in enumerate(paper.pages):
            extracted_text = page.extract_text()
            # Only append if there is actual text
            if extracted_text and extracted_text.strip():
                text_data.append({"page_number": i + 1, "text": extracted_text.strip()})
        return text_data


class DocxLoader(BaseLoader):
    def read(self):
        """
        Extract text from Word document (.docx).
        Groups paragraphs into logical pages (every 10 paragraphs = 1 page).
        """
        doc = Document(self.file_path)
        text_data = []

        # Collect all non-empty paragraphs
        paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]

        if not paragraphs:
            return text_data

        # Group paragraphs into pages (10 paragraphs per page)
        paragraphs_per_page = 10
        for i in range(0, len(paragraphs), paragraphs_per_page):
            page_paragraphs = paragraphs[i:i + paragraphs_per_page]
            page_text = "\n\n".join(page_paragraphs)

            text_data.append({
                "page_number": (i // paragraphs_per_page) + 1,
                "text": page_text
            })

        return text_data
