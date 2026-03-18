from typing import Iterable, List, Optional
import numpy as np
from tqdm import tqdm

from personalization.retrieval.preference_store.schemas import ChatTurn, MemoryCard
from personalization.feedback.schemas import TurnSample
from personalization.retrieval.pipeline import retrieve_with_rerank
from personalization.models.llm.qwen_instruct import QwenInstruct
from personalization.models.embedding.base import EmbeddingModel
from personalization.models.reranker.base import Reranker
from personalization.user_model.tensor_store import UserTensorStore

def build_turn_samples_from_sessions(
    sessions: Iterable[List[ChatTurn]],
    embed_model: EmbeddingModel,
    llm: QwenInstruct,
    reranker: Reranker,
    memory_cards: List[MemoryCard],
    memory_embeddings: np.ndarray,
    user_store: UserTensorStore,
    item_vectors: np.ndarray,
    max_samples: Optional[int] = None,
    topk_dense: int = 64,
    topk_rerank: int = 3,
) -> List[TurnSample]:
    samples = []
    
    for turns in tqdm(sessions, desc="Building TurnSamples"):
        if max_samples and len(samples) >= max_samples:
            break
            
        # Ensure sorted by turn_id
        sorted_turns = sorted(turns, key=lambda x: x.turn_id)
        
        # Iterate to find (q_t, a_t, q_{t+1})
        for i in range(len(sorted_turns)):
            if max_samples and len(samples) >= max_samples:
                break
                
            q_t = sorted_turns[i]
            if q_t.role != "user":
                continue
                
            # Find next user turn
            # Also try to find assistant response in between
            a_t_text = ""
            q_t1 = None
            
            # Look ahead
            for j in range(i + 1, len(sorted_turns)):
                next_turn = sorted_turns[j]
                if next_turn.role == "assistant" and not a_t_text:
                    a_t_text = next_turn.text
                elif next_turn.role == "user":
                    q_t1 = next_turn
                    break
            
            if not q_t1:
                # End of session or no subsequent user query
                continue
            
            # We have q_t, a_t (optional but preferred), q_t1
            # If a_t is missing, we might skip or use empty string. 
            # For RL, we usually need the answer to evaluate quality.
            # If dataset doesn't have assistant turns, we might need to generate one?
            # For now, let's proceed even if a_t is empty, or maybe require it.
            if not a_t_text:
                # Try to use LLM to generate if needed, but for offline sampling 
                # from existing chats, we prefer existing answers.
                # If using OASST1, it should have assistant turns.
                pass

            # 3. Retrieve memories for q_t
            memories_t = retrieve_with_rerank(
                user_id=q_t.user_id,
                query=q_t.text,
                embed_model=embed_model,
                reranker=reranker,
                memory_cards=memory_cards,
                memory_embeddings=memory_embeddings,
                user_store=user_store,
                item_vectors=item_vectors,
                topk_dense=topk_dense,
                topk_rerank=topk_rerank,
                beta_long=0.0,
                beta_short=0.0,
                only_own_memories=True # Assume we want user specific memories
            )
            
            # 4. Precompute embeddings
            # We can do this efficiently later or batch, but here per sample
            e_q_t = embed_model.encode([q_t.text], return_tensor=False)[0]
            e_q_t1 = embed_model.encode([q_t1.text], return_tensor=False)[0]
            
            sample = TurnSample(
                user_id=q_t.user_id,
                session_id=q_t.session_id,
                turn_id=q_t.turn_id,
                query_t=q_t.text,
                answer_t=a_t_text,
                query_t1=q_t1.text,
                memories=memories_t,
                query_embedding_t=np.array(e_q_t),
                query_embedding_t1=np.array(e_q_t1)
            )
            samples.append(sample)
            
    return samples

