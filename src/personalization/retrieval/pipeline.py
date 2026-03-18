from typing import List, Tuple
import numpy as np

from personalization.models.embedding.base import EmbeddingModel
from personalization.models.reranker.base import Reranker
from personalization.retrieval.preference_store.schemas import MemoryCard
from personalization.user_model.tensor_store import UserTensorStore, UserState
from personalization.user_model.scoring import score_with_user
from personalization.user_model.policy.reinforce import compute_policy_scores

def cosine_similarity_matrix(E: np.ndarray, e_q: np.ndarray) -> np.ndarray:
    # E: [M, d], e_q: [d]
    return np.dot(E, e_q)


def dynamic_topk_selection(
    scores: np.ndarray,
    min_k: int = 3,
    max_k: int = 8,
    score_ratio: float = 0.5,
) -> List[int]:
    """
    Dynamically select top-k indices based on score distribution.

    Strategy:
    1. Sort by score descending
    2. Compute threshold = top_score * score_ratio
    3. Select all indices with score > threshold
    4. Clamp to [min_k, max_k] range

    Args:
        scores: Array of scores (higher = better)
        min_k: Minimum number of items to select
        max_k: Maximum number of items to select
        score_ratio: Threshold ratio relative to top score

    Returns:
        List of selected indices (in descending score order)
    """
    if len(scores) == 0:
        return []

    # Sort indices by score descending
    sorted_indices = np.argsort(scores)[::-1]
    sorted_scores = scores[sorted_indices]

    # Compute threshold
    top_score = sorted_scores[0]
    threshold = top_score * score_ratio

    # Find how many pass threshold
    n_above_threshold = np.sum(sorted_scores > threshold)

    # Clamp to [min_k, max_k]
    n_select = max(min_k, min(max_k, n_above_threshold))
    n_select = min(n_select, len(scores))  # Don't exceed available

    return sorted_indices[:n_select].tolist()

def dense_topk_indices(
    query: str,
    embed_model: EmbeddingModel,
    memory_embeddings: np.ndarray,
    valid_indices: List[int] = None,
    topk: int = 64
) -> List[int]:
    """
    Return indices of topk memories based on dense embedding similarity.
    If valid_indices is provided, only search within that subset.
    """
    if valid_indices is not None and len(valid_indices) == 0:
        return []

    e_q_list = embed_model.encode([query], normalize=True, return_tensor=False)
    e_q = np.array(e_q_list[0], dtype=np.float32)
    
    # Select subset of embeddings if restricted
    if valid_indices is not None:
        # subset_embeddings = memory_embeddings[valid_indices]
        # But valid_indices might be arbitrary.
        # Efficient way: only dot product with subset
        # E_sub: [M_sub, d]
        E_sub = memory_embeddings[valid_indices]
        sims_sub = np.dot(E_sub, e_q)
        
        # Topk within subset
        k = min(topk, len(sims_sub))
        if k == 0:
            return []
            
        # argsort gives indices relative to E_sub (0..M_sub-1)
        # We need to map back to original indices
        idx_sub = np.argsort(sims_sub)[-k:][::-1]
        
        return [valid_indices[i] for i in idx_sub]
    
    # Global search
    sims = np.dot(memory_embeddings, e_q)
    k = min(topk, len(memory_embeddings))
    if k == 0:
        return []
        
    idx = np.argsort(sims)[-k:][::-1]
    return idx.tolist()

def dense_topk_indices_multi_query(
    queries: List[str],
    embed_model: EmbeddingModel,
    memory_embeddings: np.ndarray,
    valid_indices: List[int] = None,
    topk: int = 64
) -> List[int]:
    """
    Multi-query dense retrieval: embed all queries, take max similarity per memory,
    return top-k by max similarity (union effect).
    """
    if len(memory_embeddings) == 0:
        return []

    # Embed all queries at once
    e_qs = embed_model.encode(queries, normalize=True, return_tensor=False)
    e_qs = np.array(e_qs, dtype=np.float32)  # [Q, d]

    if valid_indices is not None:
        if len(valid_indices) == 0:
            return []
        E_sub = memory_embeddings[valid_indices]
        # sims: [Q, M_sub]
        sims = np.dot(e_qs, E_sub.T)
        # max across queries per memory
        max_sims = sims.max(axis=0)  # [M_sub]
        k = min(topk, len(max_sims))
        if k == 0:
            return []
        idx_sub = np.argsort(max_sims)[-k:][::-1]
        return [valid_indices[i] for i in idx_sub]

    # Global search
    # sims: [Q, M]
    sims = np.dot(e_qs, memory_embeddings.T)
    max_sims = sims.max(axis=0)  # [M]
    k = min(topk, len(max_sims))
    if k == 0:
        return []
    idx = np.argsort(max_sims)[-k:][::-1]
    return idx.tolist()


def retrieve_with_policy(
    user_id: str,
    query: str,
    embed_model: EmbeddingModel,
    reranker: Reranker,
    memory_cards: List[MemoryCard],
    memory_embeddings: np.ndarray,  # shape: [M, d]
    user_store: UserTensorStore,
    item_vectors: np.ndarray,       # shape: [M, k], v_m
    topk_dense: int = 64,
    topk_rerank: int = 8,
    beta_long: float = 0.0,
    beta_short: float = 0.0,
    tau: float = 1.0,
    only_own_memories: bool = False,
    sample: bool = False,
    queries: List[str] = None,
) -> Tuple[List[MemoryCard], np.ndarray, np.ndarray, List[int], np.ndarray]:
    """
    Returns extended info for policy update:
    (candidates, candidate_item_vectors, base_scores, chosen_indices, policy_probs)
    
    Args:
        sample: If True, use stochastic sampling from policy distribution (for training/exploration).
                If False, use deterministic top-k by policy scores (for evaluation).
    """
    # 0. Filter indices if needed
    valid_indices = None
    if only_own_memories:
        valid_indices = [i for i, card in enumerate(memory_cards) if card.user_id == user_id]
        if not valid_indices:
            return [], np.array([]), np.array([]), [], np.array([])

    # 1. Dense retrieval (multi-query if available)
    if queries and len(queries) > 1:
        dense_idx = dense_topk_indices_multi_query(
            queries,
            embed_model,
            memory_embeddings,
            valid_indices=valid_indices,
            topk=topk_dense
        )
    else:
        dense_idx = dense_topk_indices(
            query,
            embed_model,
            memory_embeddings,
            valid_indices=valid_indices,
            topk=topk_dense
        )
    # DEBUG: Check for duplicates or out of bounds
    if len(dense_idx) > 0:
        import os
        if os.getenv("RETRIEVAL_DEBUG") == "1":
            print(f"  [Pipeline] Dense Indices (Top {len(dense_idx)}): {dense_idx[:10]}...")
            print(f"  [Pipeline] Max Index: {max(dense_idx)} | Memory Size: {len(memory_cards)}")

    if not dense_idx:
        return [], np.array([]), np.array([]), [], np.array([])

    candidates = [memory_cards[i] for i in dense_idx]
    candidate_docs = [c.note_text for c in candidates]

    # 2. Rerank base score (P(yes|q,m)) - always use original query for reranking
    # Skip reranking if we have fewer candidates than topk_rerank (saves GPU memory)
    if len(candidates) <= topk_rerank:
        base_scores = np.ones(len(candidates))  # Uniform scores
    else:
        base_scores = np.array(reranker.score(query, candidate_docs))

    # 3. Policy Scoring (Softmax)
    user_state: UserState = user_store.get_state(user_id)
    candidate_vectors = item_vectors[dense_idx] # [K, k]
    
    policy_out = compute_policy_scores(
        base_scores=base_scores,
        user_state=user_state,
        item_vectors=candidate_vectors,
        beta_long=beta_long,
        beta_short=beta_short,
        tau=tau
    )
    
    # 4. Selection: Greedy (eval) or Stochastic (training)
    k = min(topk_rerank, len(policy_out.scores))
    
    if sample:
        # Stochastic sampling from policy distribution (for training/exploration)
        # Sample k indices without replacement, weighted by policy probs
        probs = policy_out.probs
        # Normalize to ensure sum to 1 (handle numerical issues)
        probs = probs / (probs.sum() + 1e-10)
        # Sample without replacement
        chosen_indices = np.random.choice(
            len(probs), size=k, replace=False, p=probs
        ).tolist()
    else:
        # Deterministic top-k by policy scores (for evaluation)
        top_indices_local = policy_out.scores.argsort()[-k:][::-1]
        chosen_indices = top_indices_local.tolist()
    
    import os
    if os.getenv("RETRIEVAL_DEBUG") == "1":
        print(f"  [Pipeline] Candidates: {len(candidates)} | Chosen Indices: {chosen_indices} | Sample: {sample}")
        
    return candidates, candidate_vectors, base_scores, chosen_indices, policy_out.probs

def retrieve_no_policy(
    user_id: str,
    query: str,
    embed_model: EmbeddingModel,
    reranker: Reranker,
    memory_cards: List[MemoryCard],
    memory_embeddings: np.ndarray,  # shape: [M, d]
    topk_dense: int = 64,
    topk_rerank: int = 8,
    only_own_memories: bool = False,
    queries: List[str] = None,
    dynamic_topk: bool = False,
    dynamic_min_k: int = 3,
    dynamic_max_k: int = 8,
    dynamic_score_ratio: float = 0.5,
) -> Tuple[List[MemoryCard], np.ndarray, np.ndarray, List[int], np.ndarray]:
    """
    Deterministic retrieval baseline (NoPersonal mode):
    - Dense retrieval -> Rerank -> Top-K (no policy sampling, no user vector influence)

    Args:
        dynamic_topk: If True, use dynamic selection based on score distribution
        dynamic_min_k: Minimum items to select (when dynamic_topk=True)
        dynamic_max_k: Maximum items to select (when dynamic_topk=True)
        dynamic_score_ratio: Threshold = top_score * ratio (when dynamic_topk=True)

    Returns same structure as retrieve_with_policy for compatibility:
    (candidates, candidate_item_vectors, base_scores, chosen_indices, rerank_scores_for_chosen)

    Note: candidate_item_vectors is empty array (not used in NoPersonal mode)
          The last return value is rerank scores instead of policy probs
    """
    # 0. Filter indices if needed
    valid_indices = None
    if only_own_memories:
        valid_indices = [i for i, card in enumerate(memory_cards) if card.user_id == user_id]
        if not valid_indices:
            return [], np.array([]), np.array([]), [], np.array([])

    # 1. Dense retrieval (multi-query if available)
    if queries and len(queries) > 1:
        dense_idx = dense_topk_indices_multi_query(
            queries,
            embed_model,
            memory_embeddings,
            valid_indices=valid_indices,
            topk=topk_dense
        )
    else:
        dense_idx = dense_topk_indices(
            query,
            embed_model,
            memory_embeddings,
            valid_indices=valid_indices,
            topk=topk_dense
        )

    if not dense_idx:
        return [], np.array([]), np.array([]), [], np.array([])

    candidates = [memory_cards[i] for i in dense_idx]
    candidate_docs = [c.note_text for c in candidates]

    # 2. Rerank base score (P(yes|q,m)) - always use original query for reranking
    max_k = dynamic_max_k if dynamic_topk else topk_rerank

    # Skip reranking if we have fewer candidates than needed
    if len(candidates) <= max_k:
        # Just return all candidates without reranking
        base_scores = np.ones(len(candidates))  # Uniform scores
        chosen_indices = list(range(len(candidates)))
    else:
        base_scores = np.array(reranker.score(query, candidate_docs))

        # 3. Selection: dynamic or fixed top-K
        if dynamic_topk:
            chosen_indices = dynamic_topk_selection(
                base_scores,
                min_k=dynamic_min_k,
                max_k=dynamic_max_k,
                score_ratio=dynamic_score_ratio,
            )
        else:
            k = min(topk_rerank, len(base_scores))
            top_indices_local = base_scores.argsort()[-k:][::-1]
            chosen_indices = top_indices_local.tolist()

    # Get scores for chosen items (for logging compatibility)
    chosen_scores = base_scores[chosen_indices]

    # Return empty item vectors (not used in NoPersonal mode)
    # Return rerank scores as the "probs" field for logging compatibility
    return candidates, np.array([]), base_scores, chosen_indices, chosen_scores


def retrieve_with_rerank(
    user_id: str,
    query: str,
    embed_model: EmbeddingModel,
    reranker: Reranker,
    memory_cards: List[MemoryCard],
    memory_embeddings: np.ndarray,  # shape: [M, d]
    user_store: UserTensorStore,
    item_vectors: np.ndarray,       # shape: [M, k], v_m
    topk_dense: int = 64,
    topk_rerank: int = 8,
    beta_long: float = 0.0,
    beta_short: float = 0.0,
    only_own_memories: bool = False,
) -> List[MemoryCard]:
    """
    Wrapper around retrieve_with_policy for standard inference.
    """
    candidates, _, _, chosen_indices, _ = retrieve_with_policy(
        user_id=user_id,
        query=query,
        embed_model=embed_model,
        reranker=reranker,
        memory_cards=memory_cards,
        memory_embeddings=memory_embeddings,
        user_store=user_store,
        item_vectors=item_vectors,
        topk_dense=topk_dense,
        topk_rerank=topk_rerank,
        beta_long=beta_long,
        beta_short=beta_short,
        tau=1.0, # Default tau
        only_own_memories=only_own_memories
    )
    
    return [candidates[i] for i in chosen_indices]


