import numpy as np
from personalization.feedback.schemas import TurnSample

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def estimate_reward(sample: TurnSample) -> float:
    """
    Return a scalar reward_hat, indicating if the previous answer was helpful.
    Range: [-1.0, 1.0] (approx)
    """
    
    # 1. Language/Topic Coherence
    if sample.query_embedding_t is None or sample.query_embedding_t1 is None:
        topic_sim = 0.5
    else:
        topic_sim = cosine_sim(sample.query_embedding_t, sample.query_embedding_t1)
        
    # 2. Negative Keywords (Complaint/Correction)
    negative_keywords = [
        "you didn't", "that's not", "incorrect", "redo", "again", "explain more", 
        "doesn't help", "wrong", "no", "not what i asked",
        "你没", "不是", "这不是", "重来", "重新", "不对", "错了", "没说清楚"
    ]
    
    # 3. Positive Keywords (Follow-up/Elaboration)
    positive_keywords = [
        "can you elaborate", "give an example", "continue", "what if", "based on that", 
        "thanks", "good", "great", "cool",
        "能不能详细一点", "举个例子", "再继续", "那如果", "接下来", "在这个基础上", "谢谢", "不错"
    ]
    
    q1_lower = sample.query_t1.lower()
    
    has_negative = any(kw in q1_lower for kw in negative_keywords)
    has_positive = any(kw in q1_lower for kw in positive_keywords)
    
    reward = 0.0
    
    if has_negative:
        reward -= 1.0
        
    if has_positive:
        # Only reward if topic similarity is decent, otherwise might be "thanks, bye" (end of session)
        # But "thanks" is good.
        reward += 0.5
        if topic_sim > 0.3:
            reward += 0.5
            
    if topic_sim < 0.2:
        # Topic shift -> previous interaction likely finished or failed. 
        # If no explicit positive/negative, assume neutral/slightly decayed.
        # If user changes topic, it often means the previous task is done (neutral/positive) 
        # OR they gave up (negative). Hard to tell. 
        # Let's dampen the reward towards 0.
        reward *= 0.5
        
    # Clip
    return max(-1.0, min(1.0, reward))

