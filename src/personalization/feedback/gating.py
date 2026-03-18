import numpy as np
from personalization.feedback.schemas import TurnSample

def cosine_sim_batch(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    # matrix: [N, d], vector: [d]
    # return: [N]
    norm_m = np.linalg.norm(matrix, axis=1)
    norm_v = np.linalg.norm(vector)
    
    # Avoid div by zero
    den = norm_m * norm_v
    den[den == 0] = 1e-9
    
    return np.dot(matrix, vector) / den

def estimate_retrieval_gating(sample: TurnSample, reward_hat: float) -> float:
    """
    Return g_t in [0,1], representing how much the reward is due to retrieval.
    """
    e_q = sample.query_embedding_t
    e_q1 = sample.query_embedding_t1
    
    if e_q is None or e_q1 is None or not sample.memories:
        return 0.5 # Neutral
        
    # We need embeddings of the memories. 
    # In a real pipeline, we might pass them in sample.memory_embeddings.
    # If missing, we can't compute sim.
    if sample.memory_embeddings is None:
        # Try to use embedding_e from memory cards if available
        # But MemoryCard.embedding_e is List[float]
        try:
            mem_embs = np.array([m.embedding_e for m in sample.memories])
            if mem_embs.shape[1] == 0: # Empty embeddings
                return 0.5
        except:
            return 0.5
    else:
        mem_embs = sample.memory_embeddings

    # Compute similarities
    # shape: [K]
    sims_q = cosine_sim_batch(mem_embs, e_q)
    sims_q1 = cosine_sim_batch(mem_embs, e_q1)
    
    s_q_max = sims_q.max() if len(sims_q) > 0 else 0
    s_q1_max = sims_q1.max() if len(sims_q1) > 0 else 0
    
    g = 0.5
    
    # Heuristics
    
    # Case A: Retrieval clearly irrelevant + bad reward
    # q_t / q_{t+1} have low similarity to memories -> likely retrieval failure (or no relevant memories)
    if reward_hat < -0.5 and s_q_max < 0.2 and s_q1_max < 0.2:
        g = 0.9 # Blame retrieval (for failing to find anything, or nothing exists)
        
    # Case B: Retrieval looks good but reward is bad
    # Memories are relevant to query, but user still unhappy -> LLM didn't use them well?
    elif reward_hat < -0.5 and s_q_max > 0.5:
        g = 0.2 # Likely LLM fault
        
    # Case C: Good reward
    # If reward is high, we assume both did okay. 
    elif reward_hat > 0.5:
        if s_q_max > 0.4:
            g = 0.6 # Retrieval helped
        else:
            g = 0.3 # LLM handled it without strong retrieval help
            
    return float(g)

