from typing import Tuple, List, Optional
import numpy as np

from personalization.retrieval.preference_store.schemas import MemoryCard
from personalization.feedback.schemas import TurnSample
from personalization.feedback.reward_model import estimate_reward
from personalization.feedback.gating import estimate_retrieval_gating
from personalization.feedback.llm_reward import (
    LLMRewardClient, LLMRewardConfig, RewardResult
)


def eval_step(
    q_t: str,
    answer_t: str,
    q_t1: str,
    memories_t: List[MemoryCard],
    query_embedding_t: Optional[np.ndarray] = None,
    query_embedding_t1: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Keyword-based evaluation (legacy).
    Given (q_t, a_t, q_{t+1}, memories), returns (reward_hat, gating_hat).
    """
    mem_embs = None
    if memories_t and memories_t[0].embedding_e:
        try:
            mem_embs = np.array([m.embedding_e for m in memories_t])
        except:
            pass

    sample = TurnSample(
        user_id="",
        session_id="",
        turn_id=0,
        query_t=q_t,
        answer_t=answer_t,
        query_t1=q_t1,
        memories=memories_t,
        query_embedding_t=query_embedding_t,
        query_embedding_t1=query_embedding_t1,
        memory_embeddings=mem_embs,
    )

    r_hat = estimate_reward(sample)
    g_hat = estimate_retrieval_gating(sample, r_hat)

    return r_hat, g_hat


async def eval_step_llm(
    q_t: str,
    answer_t: str,
    q_t1: str,
    memories_t: List[MemoryCard],
    client: LLMRewardClient,
    query_embedding_t: Optional[np.ndarray] = None,
    query_embedding_t1: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    LLM-as-judge evaluation (async).
    Returns (reward, gating) where gating=0.0 if update should be skipped.

    The gating signal is derived from the judge's confidence and label:
      - If confidence < tau_c or label == topic_shift: gating = 0.0
      - Otherwise: gating = confidence (continuous, in [tau_c, 1.0])

    This replaces the old heuristic gating with the judge's own confidence.
    """
    sample = TurnSample(
        user_id="",
        session_id="",
        turn_id=0,
        query_t=q_t,
        answer_t=answer_t,
        query_t1=q_t1,
        memories=memories_t,
        query_embedding_t=query_embedding_t,
        query_embedding_t1=query_embedding_t1,
    )

    result: RewardResult = await client.judge(sample)

    if result.should_update:
        return result.reward, result.confidence
    else:
        return 0.0, 0.0
