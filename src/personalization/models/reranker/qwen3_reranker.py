from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import Reranker
from personalization.config.settings import LocalModelsConfig
from personalization.config.registry import choose_dtype, choose_device_map

class Qwen3Reranker(Reranker):
    def __init__(self, model_path: str, device_map: str = "auto", dtype: torch.dtype = torch.bfloat16):
        # Ensure we pass trust_remote_code=True for Qwen models
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Handle specific device assignment (e.g., "cuda:0", "cuda:1")
        if device_map and device_map.startswith("cuda:"):
            # Load to CPU first, then move to specific GPU
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            self.model = self.model.to(device_map)
        else:
            # Use accelerate's auto device mapping
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
        
        self.yes_token_id = self.tokenizer("yes", add_special_tokens=False).input_ids[0]

    @classmethod
    def from_config(cls, cfg: LocalModelsConfig) -> "Qwen3Reranker":
        if not cfg.reranker or not cfg.reranker.qwen3_8b:
            raise ValueError("Reranker config for qwen3_8b is missing")
        spec = cfg.reranker.qwen3_8b
        dtype = choose_dtype(spec.dtype)
        device_map = choose_device_map(spec.device_map)
        return cls(spec.local_path, device_map=device_map, dtype=dtype)

    def _build_prompt(self, query: str, doc: str) -> str:
        return (
            "You are a reranker. "
            "Given a user query and a memory note, answer 'yes' if the note is helpful "
            "for answering the query, otherwise answer 'no'.\n\n"
            f"Query: {query}\n"
            f"Note: {doc}\n"
            "Answer with a single token: yes or no."
        )

    @torch.inference_mode()
    def score(self, query: str, docs: List[str], batch_size: int = 8, **kwargs) -> List[float]:
        scores = []
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i : i + batch_size]
            prompts = [self._build_prompt(query, d) for d in batch_docs]
            
            inputs = self.tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.model.device)
            
            outputs = self.model(**inputs)
            # Take logits of the last token
            # shape: [batch, seq_len, vocab_size]
            logits = outputs.logits
            
            # We want the logits for the token position immediately after the prompt ends.
            # But since we generated inputs directly from tokenizer(prompts), 
            # we look at the last position of the input.
            # For causal LM, we usually look at the logits of the last token 
            # to predict the *next* token (which we hope is 'yes' or 'no').
            
            # Get logits for the next token prediction (last position)
            # For each sequence in batch, select the last token's logits
            # inputs['input_ids'] shape: [B, L]
            # logits shape: [B, L, V]
            # We want logits[:, -1, :]
            
            last_token_logits = logits[:, -1, :]
            
            # Calculate log prob of 'yes'
            # We can use log_softmax over the vocab dimension
            log_probs = torch.log_softmax(last_token_logits, dim=-1)
            yes_log_probs = log_probs[:, self.yes_token_id]
            
            scores.extend(yes_log_probs.float().cpu().numpy().tolist())
            
        return scores

