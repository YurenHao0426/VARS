from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from personalization.retrieval.preference_store.schemas import ChatTurn, MemoryCard

@dataclass
class OnlineSessionState:
    user_id: str
    history: List[ChatTurn] = field(default_factory=list)
    last_query: Optional[str] = None
    last_answer: Optional[str] = None
    last_memories: List[MemoryCard] = field(default_factory=list)
    last_query_embedding: Optional[np.ndarray] = None
    last_candidate_item_vectors: Optional[np.ndarray] = None  # [K, k]
    last_policy_probs: Optional[np.ndarray] = None            # [K]
    last_chosen_indices: List[int] = field(default_factory=list)


