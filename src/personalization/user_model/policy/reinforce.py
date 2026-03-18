from typing import Sequence, List
from dataclasses import dataclass
import numpy as np

from personalization.user_model.tensor_store import UserState

@dataclass
class PolicyScores:
    scores: np.ndarray     # [K] s(q_t, m; u)
    probs: np.ndarray      # [K] π_z(m|q_t)

def compute_policy_scores(
    base_scores: np.ndarray,      # [K], from reranker
    user_state: UserState,
    item_vectors: np.ndarray,     # [K, k], v_m for the K candidates
    beta_long: float,
    beta_short: float,
    tau: float,
) -> PolicyScores:
    """
    Compute personalized scores and softmax probabilities.
    s(q_t, m; u) = s_0(q_t,m) + z_t^{(eff)}.T @ v_m
    z_t^{(eff)} = beta_long * z_long + beta_short * z_short
    """
    if len(item_vectors) == 0:
        return PolicyScores(scores=np.array([]), probs=np.array([]))

    z_eff = beta_long * user_state.z_long + beta_short * user_state.z_short
    
    # Calculate personalized term
    # item_vectors: [K, k]
    # z_eff: [k]
    # term: [K]
    personalization_term = np.dot(item_vectors, z_eff)
    
    # Total scores
    scores = base_scores + personalization_term
    
    # Softmax
    # Use exp(score/tau)
    # Subtract max for stability
    scaled_scores = scores / tau
    exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
    probs = exp_scores / np.sum(exp_scores)
    
    return PolicyScores(scores=scores, probs=probs)

def reinforce_update_user_state(
    user_state: UserState,
    item_vectors: np.ndarray,        # [K, k] for candidates
    chosen_indices: Sequence[int],   # indices of A_t in 0..K-1
    policy_probs: np.ndarray,        # [K] π_z(m|q_t)
    reward_hat: float,               # \hat r_t
    gating: float,                   # g_t
    tau: float,
    eta_long: float,
    eta_short: float,
    ema_alpha: float,
    short_decay: float,
) -> bool:
    """
    In-place update user_state.z_long / z_short / reward_ma via REINFORCE.
    Returns True if update occurred, False otherwise.
    """
    if len(chosen_indices) == 0:
        return False

    # 1. Baseline Advantage
    advantage = gating * (reward_hat - user_state.reward_ma)
    
    # Optimization: skip if advantage is negligible
    if abs(advantage) < 1e-6:
        return False

    # 2. Chosen Vector Average (v_{chosen,t})
    chosen_mask = np.zeros(len(item_vectors), dtype=np.float32)
    for idx in chosen_indices:
        idx_int = int(idx)
        if 0 <= idx_int < len(item_vectors):
            chosen_mask[idx_int] = 1.0
            
    if chosen_mask.sum() == 0:
        return False
        
    chosen_mask /= chosen_mask.sum() # Normalize to average
    v_chosen = np.dot(chosen_mask, item_vectors) # [k]

    # 3. Expected Vector (\mu_t(z))
    # policy_probs: [K]
    # item_vectors: [K, k]
    v_expect = np.dot(policy_probs, item_vectors) # [k]

    # 4. Gradient Direction
    grad = (advantage / tau) * (v_chosen - v_expect)

    # 5. Update Vectors
    user_state.z_long  += eta_long  * grad
    user_state.z_short = (1.0 - short_decay) * user_state.z_short + eta_short * grad

    # 6. Update Reward Baseline (EMA)
    user_state.reward_ma = (1.0 - ema_alpha) * user_state.reward_ma + ema_alpha * reward_hat
    
    return True

