from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Any, Dict

import yaml
from pydantic import BaseModel, Field


class ModelSpec(BaseModel):
    hf_id: str = Field(..., description="Hugging Face repository id")
    local_path: str = Field(..., description="Local directory for model weights")
    dtype: Optional[str] = Field(
        default="bfloat16", description="Preferred torch dtype: bfloat16|float16|float32"
    )
    device_map: Optional[str] = Field(default="auto", description="Device map policy")


class EmbeddingModelsConfig(BaseModel):
    qwen3: Optional[ModelSpec] = None
    nemotron: Optional[ModelSpec] = None


class RerankerModelsConfig(BaseModel):
    qwen3_8b: Optional[ModelSpec] = None


class LocalModelsConfig(BaseModel):
    llm: ModelSpec
    preference_extractor: Any # Allow flexible dict or ModelSpec for now to support map
    embedding: Optional[EmbeddingModelsConfig] = None
    reranker: Optional[RerankerModelsConfig] = None


def _resolve_config_path(env_key: str, default_rel: str) -> Path:
    value = os.getenv(env_key)
    if value:
        return Path(value).expanduser().resolve()
    # Use project root (parent of src/personalization/config) instead of cwd
    project_root = Path(__file__).parent.parent.parent.parent
    return (project_root / default_rel).resolve()


def load_local_models_config(path: Optional[str] = None) -> LocalModelsConfig:
    config_path = Path(path) if path else _resolve_config_path(
        "LOCAL_MODELS_CONFIG", "configs/local_models.yaml"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    models = raw.get("models", {})
    embedding_cfg = None
    if "embedding" in models:
        emb = models["embedding"] or {}
        # dtype/device_map are not necessary for embedders; ModelSpec still accepts them
        embedding_cfg = EmbeddingModelsConfig(
            qwen3=ModelSpec(**emb["qwen3"]) if "qwen3" in emb else None,
            nemotron=ModelSpec(**emb["nemotron"]) if "nemotron" in emb else None,
        )
    
    reranker_cfg = None
    if "reranker" in models:
        rer = models["reranker"] or {}
        reranker_cfg = RerankerModelsConfig(
            qwen3_8b=ModelSpec(**rer["qwen3_8b"]) if "qwen3_8b" in rer else None
        )

    return LocalModelsConfig(
        llm=ModelSpec(**models["llm"]),
        preference_extractor=models["preference_extractor"], # Pass raw dict/value
        embedding=embedding_cfg,
        reranker=reranker_cfg,
    )


