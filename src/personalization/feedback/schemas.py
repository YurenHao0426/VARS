from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Any
import numpy as np

from personalization.retrieval.preference_store.schemas import MemoryCard

@dataclass
class TurnSample:
    user_id: str
    session_id: str
    turn_id: int        # index of q_t within the session
    query_t: str        # q_t
    answer_t: str       # a_t
    query_t1: str       # q_{t+1}
    memories: List[MemoryCard]   # A_t

    # Optional pre-computed vectors and features
    query_embedding_t: Optional[np.ndarray] = None
    query_embedding_t1: Optional[np.ndarray] = None
    memory_embeddings: Optional[np.ndarray] = None   # corresponding e_m or v_m for memories

