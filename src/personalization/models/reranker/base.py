from typing import List, Protocol

class Reranker(Protocol):
    def score(
        self,
        query: str,
        docs: List[str],
        **kwargs,
    ) -> List[float]:
        """
        Score multiple candidate documents for the same query.
        Higher score indicates higher relevance.
        Returns a list of floats with length equal to len(docs).
        """
        ...

