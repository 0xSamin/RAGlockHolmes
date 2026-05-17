import re
from typing import List, Tuple, Set
from langchain_core.documents import Document


class ResponseVerifier:
    """
    Verifies LLM responses are grounded in retrieved context.
    Checks: citation validity, keyword overlap, number hallucination.
    """

    def __init__(self, min_keyword_overlap: float = 0.15):
        """
        Args:
            min_keyword_overlap: Minimum fraction of response keywords that must
                                 appear in retrieved context (default 15%).
        """
        # Matches: [Page 3], [Page: 3], [Pages 3-5], [doc.pdf, Page 3]
        self.citation_pattern = re.compile(
            r"\[(?:[^\],]+,\s*)?Pages?:?\s*(\d+)(?:\s*-\s*(\d+))?\]",
            re.IGNORECASE
        )
        self.number_pattern = re.compile(r"\b\d+\.?\d*\b")

        # Words to ignore in keyword overlap (stopwords + citation noise)
        self.stopwords = {
            "the", "a", "an", "is", "in", "it", "of", "to", "and", "or",
            "for", "on", "with", "that", "this", "are", "was", "were",
            "be", "been", "as", "at", "by", "from", "has", "have", "not",
            "page", "pages", "source", "sources", "context", "document"
        }
        self.min_keyword_overlap = min_keyword_overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_citations(self, llm_response: str) -> List[str]:
        """Extract all cited page numbers from an LLM response string."""
        pages = []
        for match in self.citation_pattern.finditer(llm_response):
            start, end = match.group(1), match.group(2)
            if end:  # page range like [Pages 3-5]
                pages.extend(str(p) for p in range(int(start), int(end) + 1))
            else:
                pages.append(start)
        return pages

    def verify(
        self,
        llm_response: str,
        retrieved_docs: List[Document]
    ) -> Tuple[bool, str]:
        """
        Verify that an LLM response is grounded in retrieved context.

        Runs three checks in order:
          1. Citation presence  — response must cite at least one page
          2. Citation validity  — cited pages must exist in retrieved docs
          3. Keyword grounding  — enough response keywords must appear in context
          4. Number grounding   — numbers in response must appear in context
                                  (year-like numbers and small ordinals are excluded)

        Returns:
            (is_valid, reason): is_valid=True means the response passed all checks.
        """
        if not retrieved_docs:
            return False, "No retrieved documents to verify against."

        # 1. Citation presence
        cited_pages = self.extract_citations(llm_response)
        if not cited_pages:
            return False, "No citations found in response — every claim must reference [Page X]."

        # 2. Citation validity
        available_pages: Set[str] = {
            str(doc.metadata.get("page", "?")) for doc in retrieved_docs
        }
        invalid = set(cited_pages) - available_pages
        if invalid:
            return False, (
                f"Response cites page(s) {', '.join(sorted(invalid))} "
                f"which were not retrieved (available: {', '.join(sorted(available_pages))})."
            )

        # 3. Keyword grounding
        context_text = " ".join(doc.page_content for doc in retrieved_docs).lower()
        response_keywords = self._extract_keywords(llm_response)

        if response_keywords:
            matched = sum(1 for kw in response_keywords if kw in context_text)
            overlap = matched / len(response_keywords)
            if overlap < self.min_keyword_overlap:
                return False, (
                    f"Low keyword overlap with source context ({overlap:.0%}). "
                    "Response may contain information not present in the document."
                )

        # 4. Number grounding (skip years and small ordinals)
        hallucinated = self._find_hallucinated_numbers(llm_response, context_text)
        if hallucinated:
            examples = ", ".join(list(hallucinated)[:3])
            return False, f"Numbers not found in source context: {examples}."

        return True, "Response verified — all citations valid and content grounded in context."

    def get_verification_summary(
        self,
        llm_response: str,
        retrieved_docs: List[Document]
    ) -> dict:
        """
        Returns a detailed verification report dict — useful for UI display
        or logging without blocking the response.
        """
        is_valid, reason = self.verify(llm_response, retrieved_docs)
        cited_pages = self.extract_citations(llm_response)
        available_pages = [str(doc.metadata.get("page", "?")) for doc in retrieved_docs]

        return {
            "is_valid": is_valid,
            "reason": reason,
            "cited_pages": cited_pages,
            "available_pages": sorted(set(available_pages)),
            "num_retrieved_docs": len(retrieved_docs),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_keywords(self, text: str) -> List[str]:
        """Lowercase words longer than 4 chars, excluding stopwords."""
        words = re.findall(r"\b[a-zA-Z]{5,}\b", text.lower())
        return [w for w in words if w not in self.stopwords]

    def _find_hallucinated_numbers(self, response: str, context_text: str) -> Set[str]:

        response_numbers = set(self.number_pattern.findall(response))
        context_numbers = set(self.number_pattern.findall(context_text))

        cited_page_numbers = set(self.extract_citations(response))

        hallucinated = set()
        for num in response_numbers:
            # Skip years
            if re.match(r"^(19|20)\d{2}$", num):
                continue
            # Skip small numbers (ordinals, list items, etc.)
            try:
                if float(num) <= 10:
                    continue
            except ValueError:
                continue

            if num in cited_page_numbers:
                continue

            if num not in context_numbers:
                hallucinated.add(num)

        return hallucinated
