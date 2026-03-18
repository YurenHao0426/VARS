from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import torch
import yaml

from personalization.config import settings

# Project root for resolving config paths
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Avoid circular imports by NOT importing extractors here at top level
# from personalization.models.preference_extractor.base import PreferenceExtractorBase
# from personalization.models.preference_extractor.rule_extractor import QwenRuleExtractor
# from personalization.models.preference_extractor.gpt4o_extractor import GPT4OExtractor
# from personalization.models.preference_extractor.llm_extractor import PreferenceExtractorLLM

_DTYPE_MAP: Dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

def choose_dtype(preferred: Optional[str] = None) -> torch.dtype:
    if preferred and preferred.lower() in _DTYPE_MAP:
        dt = _DTYPE_MAP[preferred.lower()]
    else:
        dt = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if dt is torch.bfloat16 and not torch.cuda.is_available():
        return torch.float32
    return dt

def choose_device_map(spec: Optional[str] = "auto") -> Any:
    return spec or "auto"

def ensure_local_path(path_str: str) -> str:
    path = Path(path_str)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return str(path)

# --- Chat Model Factory ---
def get_chat_model(name: str, device_override: Optional[str] = None):
    """
    Get a chat model by name.
    
    Args:
        name: Model name (e.g., "qwen_1_5b", "llama_8b")
        device_override: Optional device override (e.g., "cuda:2"). If None, uses config default.
    """
    from personalization.models.llm.base import ChatModel
    from personalization.models.llm.qwen_instruct import QwenInstruct
    from personalization.models.llm.llama_instruct import LlamaChatModel
    from personalization.models.llm.vllm_chat import VLLMChatModel
    
    cfg = settings.load_local_models_config()
    
    # Try to load raw config to support multi-backend map
    with open(_PROJECT_ROOT / "configs/local_models.yaml", "r") as f:
        raw_cfg = yaml.safe_load(f)
    
    models = raw_cfg.get("models", {}).get("llm", {})
    
    # If models['llm'] is a dict of configs (new style)
    if isinstance(models, dict) and "backend" in models.get(name, {}):
        spec = models[name]
        backend = spec.get("backend", "qwen")
        path = spec["path"]
        device = device_override or spec.get("device", "cuda")  # Use override if provided
        dtype = spec.get("dtype", "bfloat16")
        max_len = spec.get("max_context_length", 4096)
        
        if backend == "qwen":
            return QwenInstruct(
                model_path=path,
                device=device,
                dtype=choose_dtype(dtype), # Converts string to torch.dtype
                max_context_length=max_len
            )
        elif backend == "llama":
            return LlamaChatModel(
                model_path=path,
                device=device,
                dtype=choose_dtype(dtype), # Converts string to torch.dtype
                max_context_length=max_len
            )
        elif backend == "vllm":
            # Use vLLM HTTP API for high-throughput inference
            vllm_url = spec.get("vllm_url", "http://localhost:8003/v1")
            return VLLMChatModel(
                vllm_url=vllm_url,
                model_name=spec.get("model_name"),
                max_context_length=max_len
            )
    
    # Fallback to legacy single config
    return QwenInstruct.from_config(cfg)

def get_preference_extractor(name: Optional[str] = None):
    # Deferred imports to break circular dependency
    from personalization.models.preference_extractor.rule_extractor import QwenRuleExtractor
    from personalization.models.preference_extractor.gpt4o_extractor import GPT4OExtractor
    from personalization.models.preference_extractor.llm_extractor import PreferenceExtractorLLM
    
    cfg = settings.load_local_models_config()
    pref_cfg = cfg.preference_extractor
    
    if name is None:
        if isinstance(pref_cfg, dict) and "qwen3_0_6b_sft" in pref_cfg:
            name = "qwen3_0_6b_sft"
        else:
            name = "rule"

    if isinstance(pref_cfg, dict) and name in pref_cfg:
        spec = pref_cfg[name]
        if name == "qwen3_0_6b_sft":
            # Use QwenRuleExtractor which we have updated for SFT End-to-End logic
            return QwenRuleExtractor(
                model_path=spec["path"],
                device_map=spec.get("device", "auto"),
                dtype=choose_dtype(spec.get("dtype", "bfloat16")),
            )
        # Add 'default' handling if mapped to rule/gpt
        if name == "default":
             pass

    if name == "gpt4o":
        return GPT4OExtractor.from_config(cfg)
    elif name == "gpt5_mini":
        import os
        return GPT4OExtractor(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-5-mini")
    elif name == "rule":
        if isinstance(pref_cfg, dict):
             if "default" in pref_cfg:
                 # Manually construct to bypass ModelSpec mismatch if needed
                 spec_dict = pref_cfg["default"]
                 return QwenRuleExtractor(
                     model_path=spec_dict["local_path"],
                     dtype=choose_dtype(spec_dict.get("dtype")),
                     device_map=choose_device_map(spec_dict.get("device_map"))
                 )
        else:
            return QwenRuleExtractor.from_config(cfg)

    raise ValueError(f"Could not load preference extractor: {name}")
