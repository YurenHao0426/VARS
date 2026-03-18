import numpy as np
from .tensor_store import UserState

def score_with_user(
    base_score: float,
    user_state: UserState,
    v_m: np.ndarray,        # [k]
    beta_long: float,
    beta_short: float,
) -> float:
    """
    Personalized scoring:
    s = base_score + (beta_long * z_long + beta_short * z_short) . v_m
    Day2: beta_long = beta_short = 0 -> s == base_score
    """
    z_eff = beta_long * user_state.z_long + beta_short * user_state.z_short
    # dot product
    # Ensure shapes match
    if v_m.shape != z_eff.shape:
        # Just in case of dimension mismatch
        return float(base_score)
        
    term = np.dot(z_eff, v_m)
    return float(base_score + term)

