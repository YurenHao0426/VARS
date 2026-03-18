from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List
from personalization.retrieval.preference_store.schemas import ChatTurn, PreferenceList

class PreferenceExtractorBase(ABC):
    @abstractmethod
    def extract_turn(self, turns: List[ChatTurn]) -> PreferenceList:
        """
        Extract preferences from a window of chat turns (history + current query).
        """
        raise NotImplementedError

# Alias for backward compatibility if needed, 
# though specific extractors should inherit from PreferenceExtractorBase now.
PreferenceExtractor = PreferenceExtractorBase
