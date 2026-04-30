import re
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document

class ResponseVerifier:
    def __init__(self):
        self.citation_pattern = re.compile(r"\[Page:?\s*(\d+)\]|\[Pages:?\s*(\d+)-(\d+)\]")
        self.number_pattern = re.compile(r"\b\d+\.?\d*\b")

    def extract_citations(self, llm_response: str) -> List[str]:
        extracted_pages = []
        matches = self.citation_pattern.findall(llm_response)
        for match in matches:
            page_num, page_range_start, page_range_end = match
            if page_num:
                extracted_pages.append(page_num)
            elif page_range_start and page_range_end:
                for page in range(int(page_range_start), int(page_range_end) + 1):
                    extracted_pages.append(str(page))
        return extracted_pages