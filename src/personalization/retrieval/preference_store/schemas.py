from __future__ import annotations

from typing import List, Literal, Optional, Dict, Any

from pydantic import BaseModel, Field, confloat


class Preference(BaseModel):
    condition: str = Field(
        ..., min_length=1, max_length=128, description="When the rule applies"
    )
    action: str = Field(
        ..., min_length=1, max_length=256, description="What to do in that case"
    )
    confidence: confloat(ge=0.0, le=1.0) = Field(
        ..., description="Confidence the rule is correct"
    )


class PreferenceList(BaseModel):
    preferences: List[Preference] = Field(default_factory=list)


def preference_list_json_schema() -> dict:
    return PreferenceList.model_json_schema()


class ChatTurn(BaseModel):
    user_id: str
    session_id: str
    turn_id: int
    role: Literal["user", "assistant"]
    text: str
    timestamp: Optional[float] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class MemoryCard(BaseModel):
    card_id: str
    user_id: str
    source_session_id: str
    source_turn_ids: List[int]
    raw_queries: List[str]  # The original user utterances
    preference_list: PreferenceList
    note_text: str  # Summarized "condition: action" text
    embedding_e: List[float]  # The embedding vector
    kind: Literal["pref", "fact"] = "pref"
    is_global: bool = False  # True = always include in prompt, bypass retrieval
