from __future__ import annotations

from typing import List, Sequence

import torch
from transformers import AutoModel, AutoTokenizer

from personalization.config.registry import choose_dtype, choose_device_map
from personalization.config.settings import LocalModelsConfig
from .base import EmbeddingModel, _mean_pool, _maybe_normalize


class Qwen3Embedding8B(EmbeddingModel):
    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype,
        device_map: str = "auto",
        trust_remote_code: bool = True,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=True, trust_remote_code=trust_remote_code
        )
        
        # Handle specific device assignment (e.g., "cuda:0", "cuda:1")
        if device_map and device_map.startswith("cuda:"):
            # Load to CPU first, then move to specific GPU
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=None,  # Don't use accelerate's device_map
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=True,
            )
            self.model = self.model.to(device_map)
        else:
            # Use accelerate's auto device mapping
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=True,
            )

    @classmethod
    def from_config(cls, cfg: LocalModelsConfig) -> "Qwen3Embedding8B":
        if not cfg.embedding or not cfg.embedding.qwen3:
            raise ValueError("Embedding config for qwen3 is missing")
        spec = cfg.embedding.qwen3
        dtype = choose_dtype(spec.dtype)
        device_map = choose_device_map(spec.device_map)
        return cls(
            spec.local_path,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

    @torch.inference_mode()
    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 8,
        max_length: int = 512,
        normalize: bool = True,
        return_tensor: bool = False,
    ) -> List[List[float]] | torch.Tensor:
        device = next(self.model.parameters()).device
        outputs: List[torch.Tensor] = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            model_out = self.model(**enc, output_hidden_states=False, return_dict=True)
            pooled = _mean_pool(model_out.last_hidden_state, enc["attention_mask"])  # type: ignore[attr-defined]
            pooled = _maybe_normalize(pooled, normalize)
            outputs.append(pooled)
        emb = torch.cat(outputs, dim=0)
        if return_tensor:
            return emb
        return emb.cpu().to(torch.float32).tolist()


