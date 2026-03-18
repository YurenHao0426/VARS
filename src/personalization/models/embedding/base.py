from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Sequence

import torch


class EmbeddingModel(ABC):
    @abstractmethod
    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 8,
        max_length: int = 512,
        normalize: bool = True,
        return_tensor: bool = False,
    ) -> List[List[float]] | torch.Tensor:
        """Encode a batch of texts into dense embeddings."""
        raise NotImplementedError


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # last_hidden_state: [batch, seq_len, hidden]
    # attention_mask: [batch, seq_len]
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [b, s, 1]
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1e-6)
    return summed / counts


def _maybe_normalize(x: torch.Tensor, normalize: bool) -> torch.Tensor:
    if not normalize:
        return x
    return torch.nn.functional.normalize(x, p=2, dim=-1)


