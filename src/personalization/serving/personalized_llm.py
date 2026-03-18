#!/usr/bin/env python3
"""
Personalized LLM Interface for Evaluation.

This module provides the `PersonalizedLLM` class that wraps the entire
personalization system into a clean interface for evaluation frameworks
and user simulators.

Interface contract:
- chat(user_id, query) -> AssistantResponse: Main online interface
- reset_session(user_id): Clear session history and short-term state
- reset_user(user_id): Completely reset user (long-term, short-term, memories)
- apply_feedback(feedback): Apply external feedback for RL updates
"""

from __future__ import annotations

import os
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

# Ensure src is in path for standalone usage
_src_path = os.path.join(os.path.dirname(__file__), "../../..")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from personalization.config.settings import load_local_models_config
from personalization.config.registry import get_preference_extractor, get_chat_model
from personalization.models.embedding.qwen3_8b import Qwen3Embedding8B
from personalization.models.reranker.qwen3_reranker import Qwen3Reranker
from personalization.models.reranker.bge_reranker import BGEReranker
from personalization.user_model.tensor_store import UserTensorStore, UserState
from personalization.user_model.session_state import OnlineSessionState
from personalization.user_model.features import ItemProjection
from personalization.retrieval.preference_store.schemas import (
    MemoryCard, ChatTurn, PreferenceList, Preference
)
from personalization.retrieval.pipeline import retrieve_with_policy, retrieve_no_policy
from personalization.feedback.handlers import eval_step, eval_step_llm
from personalization.feedback.llm_reward import LLMRewardClient, LLMRewardConfig
from personalization.user_model.policy.reinforce import reinforce_update_user_state


# =============================================================================
# Data Classes for Interface
# =============================================================================

@dataclass
class UsageStats:
    """Token usage statistics from a chat completion."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str


@dataclass
class DebugInfo:
    """
    Debug information for analysis and ablation studies.
    All fields are optional - fill what you have, leave empty what you don't.
    """
    selected_memory_ids: List[str] = field(default_factory=list)
    selected_memory_notes: List[str] = field(default_factory=list)
    selected_memory_scores: List[float] = field(default_factory=list)
    user_vector_before: Optional[List[float]] = None
    user_vector_after: Optional[List[float]] = None
    extracted_preferences: List[Dict[str, Any]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssistantResponse:
    """Response from the personalized LLM chat interface."""
    answer: str
    usage: UsageStats
    debug: Optional[DebugInfo] = None


@dataclass
class Feedback:
    """
    Feedback data structure for RL updates from user simulator or judge.
    
    Attributes:
        user_id: The user this feedback is for.
        turn_id: The turn this feedback refers to (from the previous turn).
        reward: Reward scalar computed by user simulator / judge.
        gating: Gating flag (1=valid learning signal, 0=skip update).
        meta: Additional metadata for training/analysis.
    """
    user_id: str
    turn_id: int
    reward: float
    gating: float  # Can be 0.0 or 1.0, or continuous
    meta: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Internal Session State Extended
# =============================================================================

@dataclass
class _SessionContext:
    """Extended session context for evaluation tracking."""
    session_state: OnlineSessionState
    turn_counter: int = 0
    # Store info needed for apply_feedback
    pending_rl_update: Optional[Dict[str, Any]] = None


# =============================================================================
# Shared Model Singletons for Multi-threaded Efficiency
# =============================================================================

_shared_embed_model = None
_shared_reranker = None
_shared_extractor = None
_shared_models_lock = None  # Will be initialized on first use


def _get_shared_models_lock():
    """Get or create the threading lock for shared models."""
    global _shared_models_lock
    if _shared_models_lock is None:
        import threading
        _shared_models_lock = threading.Lock()
    return _shared_models_lock


def get_shared_embedding_model(model_path: str, device_map: str = "auto"):
    """Get or create shared embedding model (thread-safe singleton)."""
    global _shared_embed_model
    import torch

    lock = _get_shared_models_lock()
    with lock:
        if _shared_embed_model is None:
            print(f"[SharedModels] Loading shared embedding model on {device_map}...")
            _shared_embed_model = Qwen3Embedding8B(
                model_path=model_path,
                dtype=torch.bfloat16,
                device_map=device_map,
            )
            print("[SharedModels] Shared embedding model loaded.")
        return _shared_embed_model


def get_shared_reranker(model_path: str, device_map: str = "auto", reranker_type: str = "qwen3"):
    """Get or create shared reranker model (thread-safe singleton)."""
    global _shared_reranker
    import torch

    lock = _get_shared_models_lock()
    with lock:
        if _shared_reranker is None:
            print(f"[SharedModels] Loading shared reranker ({reranker_type}) on {device_map}...")
            if reranker_type == "bge":
                _shared_reranker = BGEReranker(
                    model_path=model_path,
                    device_map=device_map,
                    dtype=torch.float16,
                )
            else:
                _shared_reranker = Qwen3Reranker(
                    model_path=model_path,
                    device_map=device_map,
                    dtype=torch.bfloat16,
                )
            print("[SharedModels] Shared reranker model loaded.")
        return _shared_reranker


def get_shared_extractor(model_path: str, device_map: str = "auto"):
    """Get or create shared preference extractor model (thread-safe singleton)."""
    global _shared_extractor
    import torch
    from personalization.models.preference_extractor.rule_extractor import QwenRuleExtractor

    lock = _get_shared_models_lock()
    with lock:
        if _shared_extractor is None:
            print(f"[SharedModels] Loading shared preference extractor on {device_map}...")
            _shared_extractor = QwenRuleExtractor(
                model_path=model_path,
                dtype=torch.bfloat16,
                device_map=device_map,
            )
            print("[SharedModels] Shared preference extractor loaded.")
        return _shared_extractor


def clear_shared_models():
    """Free all shared singleton models to reclaim GPU memory between methods."""
    global _shared_embed_model, _shared_reranker, _shared_extractor
    import gc

    lock = _get_shared_models_lock()
    with lock:
        freed = []
        if _shared_embed_model is not None:
            freed.append("embedding")
            del _shared_embed_model
            _shared_embed_model = None
        if _shared_reranker is not None:
            freed.append("reranker")
            del _shared_reranker
            _shared_reranker = None
        if _shared_extractor is not None:
            freed.append("extractor")
            del _shared_extractor
            _shared_extractor = None

    if freed:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        print(f"[SharedModels] Cleared: {', '.join(freed)}")


# =============================================================================
# PersonalizedLLM Class
# =============================================================================

class PersonalizedLLM:
    """
    Personalized LLM wrapper for evaluation frameworks.
    
    This class provides a clean interface that accepts only (user_id, query)
    for the main chat function, while internally managing:
    - User state vectors (z_long, z_short)
    - Session history
    - Memory retrieval and policy
    - Preference extraction and storage
    - RL updates
    
    Example usage:
        llm = PersonalizedLLM()
        
        # Reset user for fresh experiment
        llm.reset_user("user_123")
        
        # Start a session
        llm.reset_session("user_123")
        
        # Chat
        response = llm.chat("user_123", "What's a good recipe for dinner?")
        print(response.answer)
        
        # Apply feedback from previous turn (from turn 2 onwards)
        llm.apply_feedback(Feedback(
            user_id="user_123",
            turn_id=0,
            reward=0.8,
            gating=1.0
        ))
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        user_store_path: str = "data/users/user_store_eval.npz",
        memory_cards_path: str = "data/corpora/memory_cards.jsonl",
        memory_embeddings_path: str = "data/corpora/memory_embeddings.npy",
        item_projection_path: str = "data/corpora/item_projection.npz",
        only_own_memories: bool = True,
        enable_preference_extraction: bool = True,
        enable_rl_updates: bool = True,
        mode: str = "full",  # "full", "nopersonal", or "vanilla"
        eval_mode: bool = True,  # True = greedy selection, False = stochastic sampling
        device_assignment: Optional[Dict[str, str]] = None,  # Multi-GPU support
        llm_name: Optional[str] = None,  # Override LLM name (e.g., "llama_8b_vllm" for vLLM)
        use_shared_models: bool = False,  # Use shared singleton models for multi-threaded efficiency
        reranker_type: str = "qwen3",  # "qwen3" (8B) or "bge" (278M)
        best_of_n: int = 1,  # Generate N responses and pick best (for RAG methods)
        reward_mode: str = "keyword",  # "keyword", "llm" (GPT-4o-mini), or "llm_local" (local vLLM)
        llm_reward_config: Optional["LLMRewardConfig"] = None,  # Config for LLM judge
        reward_vllm_url: Optional[str] = None,  # vLLM URL for local reward model (when reward_mode="llm_local")
        enable_query_transform: bool = False,  # Transform queries for better retrieval matching
        enable_global_preferences: bool = False,  # Separate global prefs that bypass retrieval
        dynamic_topk: bool = False,  # Use dynamic topk based on rerank scores
        dynamic_min_k: int = 3,  # Min preferences for dynamic topk
        dynamic_max_k: int = 8,  # Max preferences for dynamic topk
        dynamic_score_ratio: float = 0.5,  # Threshold = top_score * ratio
        eta_long: float = None,  # Override RL learning rate for z_long
        eta_short: float = None,  # Override RL learning rate for z_short
        enable_preference_consolidation: bool = False,  # Consolidate preferences at session end
        consolidation_threshold: int = 5,  # Min preferences before consolidation
        enable_preference_rewrite: bool = False,  # Use LLM to rewrite/merge retrieved preferences
    ):
        """
        Initialize the PersonalizedLLM.
        
        Args:
            config_path: Path to config file. If None, uses default locations.
            user_store_path: Path to persist user state vectors.
            memory_cards_path: Path to memory cards JSONL file.
            memory_embeddings_path: Path to memory embeddings numpy file.
            item_projection_path: Path to item projection (PCA) file.
            only_own_memories: If True, only retrieve user's own memories (strict privacy).
            enable_preference_extraction: If True, extract preferences from user turns.
            enable_rl_updates: If True, apply RL updates via apply_feedback.
            mode: "full" for full personalization, "nopersonal" for baseline (no user vector influence),
                  "vanilla" for pure LLM without any memory retrieval or preference extraction.
            eval_mode: If True, use greedy/deterministic selection (for evaluation).
                       If False, use stochastic sampling (for training/exploration).
            device_assignment: Optional dict to assign models to specific GPUs.
                Example: {"embed": "cuda:0", "reranker": "cuda:1", "chat": "cuda:2", "extractor": "cuda:3"}
                If None, uses "auto" for all models.
            use_shared_models: If True, use shared singleton models for embedding and reranker.
                This is essential for multi-threaded/parallel profile processing to avoid
                loading duplicate models. When enabled, the first thread loads the models,
                and subsequent threads reuse the shared instances.
        """
        self.only_own_memories = only_own_memories
        self.use_shared_models = use_shared_models
        self.enable_preference_extraction = enable_preference_extraction
        self.enable_rl_updates = enable_rl_updates
        self.mode = mode  # "full" or "nopersonal"
        self.eval_mode = eval_mode  # True = greedy, False = sample
        self.reranker_type = reranker_type  # "qwen3" or "bge"
        self.best_of_n = best_of_n  # Generate N responses and pick best
        self.reward_mode = reward_mode  # "keyword", "llm", or "llm_local"
        self.enable_query_transform = enable_query_transform
        self.enable_global_preferences = enable_global_preferences
        self.enable_preference_consolidation = enable_preference_consolidation
        self.consolidation_threshold = consolidation_threshold
        self.enable_preference_rewrite = enable_preference_rewrite

        # Initialize LLM reward client if using LLM judge
        self._llm_reward_client = None  # Can be LLMRewardClient or LocalLLMRewardClient
        if reward_mode == "llm":
            self._llm_reward_client = LLMRewardClient(llm_reward_config or LLMRewardConfig())
        elif reward_mode == "llm_local":
            from personalization.feedback.local_llm_reward import (
                LocalLLMRewardClient,
                LocalLLMRewardConfig,
            )
            local_config = LocalLLMRewardConfig(
                vllm_url=reward_vllm_url or "http://localhost:8005/v1",
            )
            self._llm_reward_client = LocalLLMRewardClient(local_config)
        
        # Multi-GPU device assignment
        self._device_assignment = device_assignment or {
            "embed": "auto",
            "reranker": "auto",
            "chat": "auto",
            "extractor": "auto",
        }
        
        # Paths
        self._memory_cards_path = memory_cards_path
        self._memory_embeddings_path = memory_embeddings_path
        self._item_projection_path = item_projection_path
        
        # RL Configuration
        # Note: beta/eta increased for more significant z_u updates
        self._rl_cfg = {
            "item_dim": 256,
            "beta_long": 2.0,    # Increased from 0.1 for stronger personalization
            "beta_short": 5.0,   # Increased from 0.3
            "tau": 1.0,
            "eta_long": eta_long if eta_long is not None else 0.01,
            "eta_short": eta_short if eta_short is not None else 0.05,
            "ema_alpha": 0.05,
            "short_decay": 0.1,
            "dense_topk": 64,
            "rerank_topk": 5,
            "max_new_tokens": 512,
            # Dynamic topk settings
            "dynamic_topk": dynamic_topk,
            "dynamic_min_k": dynamic_min_k,
            "dynamic_max_k": dynamic_max_k,
            "dynamic_score_ratio": dynamic_score_ratio,
        }
        
        # Store llm_name before loading config (needed in _load_config)
        self._llm_name_override = llm_name

        # Load config and override RL params if available
        self._load_config(config_path)
        
        # Load models
        print("[PersonalizedLLM] Loading models...")
        self._load_models()
        
        # Load memory store
        print("[PersonalizedLLM] Loading memory store...")
        self._load_memory_store()
        
        # Initialize user store
        self._user_store = UserTensorStore(
            k=self._rl_cfg["item_dim"],
            path=user_store_path,
        )
        
        # Session contexts per user (in-memory)
        self._sessions: Dict[str, _SessionContext] = {}
        
        print("[PersonalizedLLM] Initialization complete.")
    
    def _load_config(self, config_path: Optional[str]):
        """Load configuration from yaml files."""
        self._cfg = load_local_models_config()
        
        # Try to load user_model.yaml for RL params
        if config_path is None:
            config_path = "configs/user_model.yaml"
        
        self._llm_name = self._llm_name_override or "qwen_1_5b"  # Default, can be overridden

        try:
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    user_cfg = yaml.safe_load(f)
                    if user_cfg:
                        # Override RL params if present
                        for key in self._rl_cfg:
                            if key in user_cfg:
                                self._rl_cfg[key] = user_cfg[key]
                        # LLM name (only from config if not already set via parameter)
                        if self._llm_name_override is None and "llm_name" in user_cfg:
                            self._llm_name = user_cfg["llm_name"]
        except Exception as e:
            print(f"[PersonalizedLLM] Warning: Failed to load config: {e}")
    
    def _load_models(self):
        """Load all ML models with optional multi-GPU assignment."""
        import torch

        # Report GPU availability (only once, not for shared model instances)
        if not self.use_shared_models:
            num_gpus = torch.cuda.device_count()
            print(f"[PersonalizedLLM] Available GPUs: {num_gpus}")
            for i in range(num_gpus):
                mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({mem:.1f}GB)")

        embed_device = self._device_assignment.get("embed", "auto")
        reranker_device = self._device_assignment.get("reranker", "auto")
        chat_device = self._device_assignment.get("chat", "auto")
        extractor_device = self._device_assignment.get("extractor", "auto")

        # Embedding model - only load for modes that use RAG retrieval
        # Vanilla and contextual modes don't need embedding/reranker
        needs_retrieval = self.mode not in ("vanilla", "contextual")

        if needs_retrieval:
            if self.use_shared_models:
                print(f"[PersonalizedLLM] Using shared embedding model...")
                self._embed_model = get_shared_embedding_model(
                    model_path=self._cfg.embedding.qwen3.local_path,
                    device_map=embed_device,
                )
            else:
                print(f"[PersonalizedLLM] Loading Embedding model on {embed_device}...")
                self._embed_model = Qwen3Embedding8B(
                    model_path=self._cfg.embedding.qwen3.local_path,
                    dtype=torch.bfloat16,
                    device_map=embed_device,
                )
        else:
            print(f"[PersonalizedLLM] Skipping embedding model (not needed for {self.mode} mode)")
            self._embed_model = None

        # Reranker - only load for modes that use RAG retrieval
        # Support both qwen3 (8B) and bge (278M) rerankers
        if needs_retrieval:
            if self.reranker_type == "bge":
                reranker_path = getattr(self._cfg.reranker, "bge_base", None)
                reranker_path = reranker_path.local_path if reranker_path else "BAAI/bge-reranker-base"
            else:
                reranker_path = self._cfg.reranker.qwen3_8b.local_path

            if self.use_shared_models:
                print(f"[PersonalizedLLM] Using shared reranker model ({self.reranker_type})...")
                self._reranker = get_shared_reranker(
                    model_path=reranker_path,
                    device_map=reranker_device,
                    reranker_type=self.reranker_type,
                )
            else:
                print(f"[PersonalizedLLM] Loading Reranker ({self.reranker_type}) on {reranker_device}...")
                if self.reranker_type == "bge":
                    self._reranker = BGEReranker(
                        model_path=reranker_path,
                        device_map=reranker_device,
                        dtype=torch.float16,
                    )
                else:
                    self._reranker = Qwen3Reranker(
                        model_path=reranker_path,
                        device_map=reranker_device,
                        dtype=torch.bfloat16,
                    )
        else:
            print(f"[PersonalizedLLM] Skipping reranker (not needed for {self.mode} mode)")
            self._reranker = None

        # Chat model (via registry for backend switching)
        print(f"[PersonalizedLLM] Loading ChatModel: {self._llm_name} on {chat_device}...")
        # Pass device override if specified (not "auto")
        device_for_chat = chat_device if chat_device != "auto" else None
        self._chat_model = get_chat_model(self._llm_name, device_override=device_for_chat)

        # Preference extractor - use shared singleton if enabled
        if self.enable_preference_extraction:
            extractor_name = "qwen3_0_6b_sft"
            if self.use_shared_models:
                print(f"[PersonalizedLLM] Using shared preference extractor...")
                try:
                    extractor_path = self._cfg.preference_extractor.get("qwen3_0_6b_sft", {}).get("path", None)
                    if extractor_path:
                        self._extractor = get_shared_extractor(
                            model_path=extractor_path,
                            device_map=extractor_device,
                        )
                    else:
                        print(f"[PersonalizedLLM] Extractor path not found, using rule-based.")
                        self._extractor = get_preference_extractor("rule")
                except Exception as e:
                    print(f"[PersonalizedLLM] Warning: Failed to load shared extractor: {e}. Trying fallbacks...")
                    try:
                        self._extractor = get_preference_extractor("rule")
                    except Exception as e2:
                        print(f"[PersonalizedLLM] Rule extractor also failed: {e2}. Using GPT-5-mini extractor.")
                        self._extractor = get_preference_extractor("gpt5_mini")
            else:
                print(f"[PersonalizedLLM] Loading extractor: {extractor_name} on {extractor_device}...")
                try:
                    self._extractor = get_preference_extractor(extractor_name)
                except Exception as e:
                    print(f"[PersonalizedLLM] Warning: Failed to load {extractor_name}: {e}. Trying fallbacks...")
                    try:
                        self._extractor = get_preference_extractor("rule")
                    except Exception as e2:
                        print(f"[PersonalizedLLM] Rule extractor also failed: {e2}. Using GPT-5-mini extractor.")
                        self._extractor = get_preference_extractor("gpt5_mini")
        else:
            print("[PersonalizedLLM] Preference extraction disabled, skipping extractor.")
            self._extractor = None
    
    def _load_memory_store(self):
        """Load memory cards and embeddings."""
        if not os.path.exists(self._memory_cards_path):
            print(f"[PersonalizedLLM] Warning: Memory cards not found at {self._memory_cards_path}")
            self._memory_cards: List[MemoryCard] = []
            self._memory_embeddings = np.zeros((0, 4096), dtype=np.float32)
            self._item_vectors = np.zeros((0, self._rl_cfg["item_dim"]), dtype=np.float32)
            # Create default projection (truncation to first k dims) so preferences can be added
            k = self._rl_cfg["item_dim"]
            d = 4096
            P = np.zeros((k, d), dtype=np.float32)
            P[:, :k] = np.eye(k, dtype=np.float32)
            self._projection = ItemProjection(P=P, mean=np.zeros(d, dtype=np.float32))
            print(f"[PersonalizedLLM] Created default projection (truncation, k={k})")
            return
        
        # Load cards
        self._memory_cards = []
        with open(self._memory_cards_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._memory_cards.append(MemoryCard.model_validate_json(line))
        
        # Load embeddings
        if os.path.exists(self._memory_embeddings_path):
            self._memory_embeddings = np.load(self._memory_embeddings_path)
        else:
            self._memory_embeddings = np.zeros((len(self._memory_cards), 4096), dtype=np.float32)
        
        # Load projection
        if os.path.exists(self._item_projection_path):
            proj_data = np.load(self._item_projection_path)
            self._projection = ItemProjection(P=proj_data["P"], mean=proj_data["mean"])
            self._item_vectors = proj_data["V"]
        else:
            # Create default projection so preferences can still be added
            k = self._rl_cfg["item_dim"]
            d = 4096
            P = np.zeros((k, d), dtype=np.float32)
            P[:, :k] = np.eye(k, dtype=np.float32)
            self._projection = ItemProjection(P=P, mean=np.zeros(d, dtype=np.float32))
            self._item_vectors = np.zeros((len(self._memory_cards), self._rl_cfg["item_dim"]), dtype=np.float32)
            print(f"[PersonalizedLLM] Created default projection (truncation, k={k})")
        
        print(f"[PersonalizedLLM] Loaded {len(self._memory_cards)} memory cards.")
    
    def _get_or_create_session(self, user_id: str) -> _SessionContext:
        """Get or create session context for a user."""
        if user_id not in self._sessions:
            self._sessions[user_id] = _SessionContext(
                session_state=OnlineSessionState(user_id=user_id),
                turn_counter=0,
            )
        return self._sessions[user_id]
    
    def _build_chat_turn(self, user_id: str, text: str, role: str, turn_id: int) -> ChatTurn:
        """Build a ChatTurn object."""
        return ChatTurn(
            user_id=user_id,
            session_id=f"eval_session_{user_id}",
            turn_id=turn_id,
            role=role,
            text=text,
            meta={"source": "eval"}
        )
    
    def _count_tokens(self, text: str) -> int:
        """Estimate token count using the tokenizer."""
        try:
            # Use the chat model's tokenizer if available
            if hasattr(self._chat_model, 'tokenizer'):
                return len(self._chat_model.tokenizer.encode(text))
            else:
                # Rough estimate: ~4 chars per token
                return len(text) // 4
        except Exception:
            return len(text) // 4
    
    # Task type keywords for query transformation
    _TASK_KEYWORDS = {
        "math": ["solve", "calculate", "integral", "equation", "proof", "derivative",
                 "math", "algebra", "geometry", "trigonometry", "calculus", "arithmetic",
                 "formula", "compute", "evaluate", "simplify", "factor", "graph"],
        "coding": ["code", "program", "function", "implement", "debug", "python", "java",
                   "javascript", "algorithm", "class", "method", "bug", "error", "compile",
                   "script", "html", "css", "sql", "api", "library", "framework"],
        "writing": ["write", "essay", "paragraph", "summarize", "draft", "compose",
                    "article", "story", "letter", "email", "report", "review", "edit",
                    "rewrite", "paraphrase", "outline"],
        "explanation": ["explain", "what is", "how does", "why", "describe", "define",
                       "meaning", "concept", "difference between", "compare", "contrast"],
    }

    def _transform_query_for_retrieval(self, query: str) -> List[str]:
        """
        Transform raw user query into multiple retrieval queries to bridge
        the semantic gap between task queries and preference descriptions.

        Returns [original_query, transformed_query] or [original_query] if
        no task type detected.
        """
        import re
        query_lower = query.lower()
        detected_types = []
        for task_type, keywords in self._TASK_KEYWORDS.items():
            for kw in keywords:
                # Use word boundary matching to avoid false positives
                # e.g., "api" should not match "capital"
                if re.search(r'\b' + re.escape(kw) + r'\b', query_lower):
                    detected_types.append(task_type)
                    break

        if not detected_types:
            return [query]

        # Use first detected type (most specific match)
        task_type = detected_types[0]
        transformed = f"user preferences for {task_type} tasks: {query}"
        return [query, transformed]

    # Patterns indicating a global/universal preference condition
    _GLOBAL_PATTERNS = ["general", "any", "always", "all ", "every", "regardless",
                        "any task", "any topic", "any question", "all tasks", "all topics"]

    # Domain-specific terms that indicate a conditional preference
    _DOMAIN_TERMS = ["math", "code", "coding", "program", "writing", "essay", "science",
                     "history", "language", "physics", "chemistry", "biology", "literature",
                     "creative", "technical", "formal", "informal", "academic", "casual"]

    def _classify_preference_scope(self, condition: str) -> bool:
        """
        Classify whether a preference condition is global (always applicable)
        or conditional (task-specific).

        Returns True if global, False if conditional.
        """
        cond_lower = condition.lower().strip()

        # Check for explicit global patterns
        for pattern in self._GLOBAL_PATTERNS:
            if pattern in cond_lower:
                return True

        # Very short/vague conditions with no domain terms are likely global
        words = cond_lower.split()
        if len(words) <= 2:
            has_domain = any(term in cond_lower for term in self._DOMAIN_TERMS)
            if not has_domain:
                return True

        return False

    # Rewrite prompt for merging retrieved preferences
    _REWRITE_PROMPT = """You are helping to prepare user preferences for an AI assistant.

The user is asking: {query}

Retrieved preferences about this user:
{preferences}

Task: Create a concise preference summary that the assistant MUST follow.

Rules:
1. PRESERVE all specific formatting requirements exactly (e.g., "type hints", "snake_case", "code fence with language")
2. PRESERVE all structural requirements (e.g., "numbered steps", "bullet points", "answer first then explanation")
3. Only MERGE preferences that are truly redundant (saying the same thing differently)
4. Output as a short bulleted list if there are multiple distinct requirements
5. Keep each point actionable and specific - NO vague generalizations like "follow best practices"

Example input:
- Include type hints in Python code
- Use snake_case for variable names
- When explaining, use numbered steps

Example output:
- Include type hints
- Use snake_case for variables
- Use numbered steps for explanations

If no preferences are relevant to this query type, output: "No specific preferences apply."

Preference summary:"""

    def _rewrite_preferences(self, memory_notes: List[str], query: str) -> List[str]:
        """
        Use LLM to rewrite/merge multiple retrieved preferences into concise instructions.

        This is similar to Reflection's proper_scaffolding but focuses on merging
        rather than just filtering.

        Args:
            memory_notes: List of retrieved preference notes
            query: Current user query

        Returns:
            List with single rewritten instruction (or original if rewrite fails/disabled)
        """
        if not memory_notes or len(memory_notes) <= 1:
            return memory_notes

        try:
            import requests

            # Format preferences for prompt
            prefs_text = "\n".join(f"- {note}" for note in memory_notes)
            prompt = self._REWRITE_PROMPT.format(query=query[:200], preferences=prefs_text)

            # Direct vLLM API call (simpler than going through chat model)
            messages = [{"role": "user", "content": prompt}]
            payload = {
                "model": self._chat_model.model_name,
                "messages": messages,
                "max_tokens": 150,
                "temperature": 0.3,  # Lower temperature for more consistent output
            }

            response = requests.post(
                f"{self._chat_model.vllm_url}/chat/completions",
                json=payload,
                timeout=30
            )

            if response.status_code != 200:
                print(f"[REWRITE] API error {response.status_code}, keeping original notes")
                return memory_notes

            result = response.json()
            rewritten = result["choices"][0]["message"]["content"].strip().strip('"')

            # Validate response
            if rewritten and len(rewritten) > 10 and "No specific preferences" not in rewritten:
                print(f"[REWRITE] {len(memory_notes)} notes → 1 merged instruction")
                return [rewritten]
            else:
                print(f"[REWRITE] Kept original {len(memory_notes)} notes (no valid merge)")
                return memory_notes

        except Exception as e:
            print(f"[REWRITE] Failed: {e}, keeping original notes")
            return memory_notes

    # Consolidation prompt for session-end preference merging
    _CONSOLIDATION_PROMPT = """You are analyzing user preferences extracted from conversations.

Current preferences for this user:
{preferences}

Task: Consolidate these preferences into a cleaner, more organized set by:
1. MERGE similar preferences (e.g., "use bullet points" + "format with bullets" → single preference)
2. REMOVE redundant or contradictory preferences (keep the more specific one)
3. PRESERVE all unique, meaningful preferences
4. Keep the same "When [condition], [action]." format

Output ONLY the consolidated preferences, one per line, in this exact format:
When [condition], [action].

Do not add explanations or commentary. Just output the preference lines."""

    def consolidate_user_preferences(self, user_id: str) -> int:
        """
        Consolidate user preferences at session end using LLM.

        Merges similar preferences, removes redundancy, and creates cleaner
        preference descriptions. Only runs if user has enough preferences.

        Args:
            user_id: The user whose preferences to consolidate.

        Returns:
            Number of preferences after consolidation (0 if skipped).
        """
        if not self.enable_preference_consolidation:
            return 0

        # Get user's memory cards
        user_cards = [c for c in self._memory_cards if c.user_id == user_id]

        if len(user_cards) < self.consolidation_threshold:
            return len(user_cards)

        # Build preference list for prompt
        pref_lines = [card.note_text for card in user_cards]
        preferences_text = "\n".join(f"- {p}" for p in pref_lines)

        # Call LLM for consolidation
        prompt = self._CONSOLIDATION_PROMPT.format(preferences=preferences_text)
        messages = [{"role": "user", "content": prompt}]

        try:
            result = self._chat_model.answer(messages, max_new_tokens=512)
            consolidated_text = result.get("content", "").strip()

            if not consolidated_text:
                return len(user_cards)

            # Parse consolidated preferences
            new_prefs = []
            for line in consolidated_text.split("\n"):
                line = line.strip()
                if not line or not line.startswith("When "):
                    continue
                # Parse "When [condition], [action]."
                if ", " in line:
                    parts = line.split(", ", 1)
                    condition = parts[0].replace("When ", "").strip()
                    action = parts[1].rstrip(".").strip()
                    if condition and action:
                        new_prefs.append({
                            "condition": condition,
                            "action": action,
                            "is_global": self._classify_preference_scope(condition) if self.enable_global_preferences else False,
                        })

            if not new_prefs:
                return len(user_cards)

            # Remove old cards for this user
            keep_indices = [i for i, c in enumerate(self._memory_cards) if c.user_id != user_id]
            self._memory_cards = [self._memory_cards[i] for i in keep_indices]
            if len(keep_indices) > 0 and len(self._memory_embeddings) > 0:
                self._memory_embeddings = self._memory_embeddings[keep_indices]
                self._item_vectors = self._item_vectors[keep_indices]
            else:
                embed_dim = self._memory_embeddings.shape[1] if len(self._memory_embeddings) > 0 else 4096
                self._memory_embeddings = np.zeros((0, embed_dim), dtype=np.float32)
                self._item_vectors = np.zeros((0, self._rl_cfg["item_dim"]), dtype=np.float32)

            # Add consolidated preferences
            for pref in new_prefs:
                note_text = f"When {pref['condition']}, {pref['action']}."

                # Compute embedding
                e_note = self._embed_model.encode([note_text], normalize=True, return_tensor=False)[0]
                v_note = self._projection.transform_vector(np.array(e_note))

                # Create card
                card = MemoryCard(
                    card_id=str(uuid.uuid4()),
                    user_id=user_id,
                    source_session_id=f"consolidated_{user_id}",
                    source_turn_ids=[],
                    raw_queries=[],
                    preference_list=PreferenceList(preferences=[
                        Preference(condition=pref["condition"], action=pref["action"], confidence=1.0)
                    ]),
                    note_text=note_text,
                    embedding_e=list(e_note),
                    kind="pref",
                    is_global=pref["is_global"],
                )

                self._memory_cards.append(card)
                self._memory_embeddings = np.vstack([self._memory_embeddings, np.array([e_note])])
                self._item_vectors = np.vstack([self._item_vectors, np.array([v_note])])

            print(f"[PersonalizedLLM] Consolidated {len(user_cards)} → {len(new_prefs)} preferences for user {user_id}")
            return len(new_prefs)

        except Exception as e:
            print(f"[PersonalizedLLM] Consolidation failed for user {user_id}: {e}")
            return len(user_cards)

    def _add_preferences_as_memory(
        self,
        prefs: PreferenceList,
        query: str,
        user_id: str,
        turn_id: int,
    ) -> List[Dict[str, Any]]:
        """
        Add extracted preferences as new memory cards.
        Returns list of preference dicts for debug info.
        """
        extracted = []

        if not prefs.preferences or self._projection is None:
            return extracted

        for pref in prefs.preferences:
            note_text = f"When {pref.condition}, {pref.action}."

            # Record for debug
            extracted.append({
                "condition": pref.condition,
                "action": pref.action,
                "confidence": pref.confidence,
            })

            # Deduplication check
            is_duplicate = any(
                card.user_id == user_id and card.note_text == note_text
                for card in self._memory_cards
            )

            if is_duplicate:
                continue

            # Compute embedding from note_text (NOT query) for proper semantic retrieval
            # This ensures retrieval query "solve math problem" matches stored "When math problems..."
            e_note = self._embed_model.encode([note_text], normalize=True, return_tensor=False)[0]
            v_note = self._projection.transform_vector(np.array(e_note))

            # Classify as global or conditional
            is_global = self._classify_preference_scope(pref.condition) if self.enable_global_preferences else False

            # Create new memory card
            card = MemoryCard(
                card_id=str(uuid.uuid4()),
                user_id=user_id,
                source_session_id=f"eval_session_{user_id}",
                source_turn_ids=[turn_id],
                raw_queries=[query],
                preference_list=PreferenceList(preferences=[pref]),
                note_text=note_text,
                embedding_e=list(e_note),
                kind="pref",
                is_global=is_global,
            )

            # Add to memory store
            self._memory_cards.append(card)
            self._memory_embeddings = np.vstack([self._memory_embeddings, np.array([e_note])])
            self._item_vectors = np.vstack([self._item_vectors, np.array([v_note])])
        
        return extracted

    def _score_response(self, response: str) -> float:
        """
        Score a response for best-of-N selection.

        Higher score = better response. Scoring heuristics:
        1. Length: Longer responses typically have more substance
        2. Solution indicators: Contains formulas, steps, answers
        3. Proactivity: Doesn't end with just a question

        Returns:
            Float score (higher is better)
        """
        score = 0.0
        response_lower = response.lower()

        # Length score (normalized, cap at 1000 chars)
        score += min(len(response), 1000) / 1000 * 3.0

        # Solution indicators (+1 each, max 5)
        solution_indicators = ['=', 'step', 'answer', 'formula', 'result', 'therefore', 'solution']
        indicator_count = sum(1 for ind in solution_indicators if ind in response_lower)
        score += min(indicator_count, 5) * 0.5

        # Structured content (+1 for numbered/bulleted lists)
        if any(marker in response for marker in ['1.', '2.', '- ', '* ', '##']):
            score += 1.0

        # Penalty for ending with question (passive behavior)
        # Check last 100 chars for question marks
        if '?' in response[-100:]:
            score -= 1.5

        # Bonus for providing concrete values/numbers
        import re
        numbers = re.findall(r'\d+\.?\d*', response)
        if len(numbers) >= 3:
            score += 1.0

        return score

    # =========================================================================
    # Public Interface
    # =========================================================================

    def chat(self, user_id: str, query: str) -> AssistantResponse:
        """
        Main online chat interface.
        
        Args:
            user_id: Unique identifier for the user.
            query: Current user query/message.
        
        Returns:
            AssistantResponse containing the answer, usage stats, and debug info.
        
        Notes:
            - Internally manages user state, session history, memory retrieval
            - After this call, you can call apply_feedback() with the turn's feedback
        """
        ctx = self._get_or_create_session(user_id)
        session = ctx.session_state
        user_state = self._user_store.get_state(user_id)

        # Record user vector before for debug
        z_long_before = user_state.z_long.copy().tolist()
        z_short_before = user_state.z_short.copy().tolist()

        # Add user turn to history
        user_turn = self._build_chat_turn(user_id, query, "user", ctx.turn_counter)
        session.history.append(user_turn)

        # Vanilla mode: pure LLM without any memory or preference extraction
        if self.mode == "vanilla":
            # Skip embedding, preference extraction, and memory retrieval entirely
            e_q_t = np.zeros(4096, dtype=np.float32)  # Placeholder for vanilla mode
            extracted_prefs = []
            candidates = []
            cand_item_vecs = np.array([])
            base_scores = np.array([])
            chosen_indices = []
            probs = np.array([])
            memories_t = []
            memory_notes = []
        else:
            # Compute query embedding (only needed for non-vanilla modes)
            # Explicitly normalize for consistent cosine similarity with stored embeddings
            embed_result = self._embed_model.encode([query], normalize=True, return_tensor=False)
            if embed_result is None or len(embed_result) == 0:
                raise RuntimeError(f"Embedding model returned empty result for query: {query[:100]}")
            e_q_t = np.array(embed_result[0])

            # Store pending RL update info from last turn (for apply_feedback)
            if session.last_query is not None and self.enable_rl_updates:
                ctx.pending_rl_update = {
                    "last_query": session.last_query,
                    "last_answer": session.last_answer,
                    "last_memories": session.last_memories,
                    "last_query_embedding": session.last_query_embedding,
                    "current_query_embedding": e_q_t,
                    "last_candidate_item_vectors": session.last_candidate_item_vectors,
                    "last_policy_probs": session.last_policy_probs,
                    "last_chosen_indices": session.last_chosen_indices,
                }

                # Auto-compute reward via LLM judge if enabled
                if self._llm_reward_client is not None:
                    import asyncio
                    try:
                        reward, gating = asyncio.run(eval_step_llm(
                            q_t=session.last_query,
                            answer_t=session.last_answer,
                            q_t1=query,
                            memories_t=session.last_memories or [],
                            client=self._llm_reward_client,
                        ))
                        if gating > 0.0:
                            self.apply_feedback(Feedback(
                                user_id=user_id,
                                turn_id=ctx.turn_counter - 1,
                                reward=reward,
                                gating=gating,
                            ))
                    except Exception as e:
                        # Graceful fallback: skip RL update if judge fails
                        print(f"[LLM-Reward] Judge call failed, skipping update: {e}")

            # Extract preferences from conversation (if enabled)
            # extract_turn processes only the last user turn - efficient since called each turn
            # Preferences accumulate in _memory_cards across turns (dedup prevents duplicates)
            extracted_prefs = []
            if self.enable_preference_extraction:
                prefs = self._extractor.extract_turn(session.history)
                if prefs.preferences:
                    print(f"[DEBUG] Extracted {len(prefs.preferences)} prefs from history (len={len(session.history)})")
                extracted_prefs = self._add_preferences_as_memory(
                    prefs, query, user_id, ctx.turn_counter
                )
                if extracted_prefs:
                    print(f"[DEBUG] Added {len(extracted_prefs)} to memory. Total cards: {len(self._memory_cards)}")
            
            # Separate global preferences (bypass retrieval) from conditional ones
            global_notes = []
            retrieval_cards = self._memory_cards
            retrieval_embeddings = self._memory_embeddings
            retrieval_item_vectors = self._item_vectors
            if self.enable_global_preferences:
                global_cards = [c for c in self._memory_cards if c.is_global and c.user_id == user_id]
                global_notes = [c.note_text for c in global_cards[:10]]  # Cap at 10
                # Filter out global cards for retrieval
                cond_indices = [i for i, c in enumerate(self._memory_cards) if not c.is_global]
                if cond_indices:
                    retrieval_cards = [self._memory_cards[i] for i in cond_indices]
                    retrieval_embeddings = self._memory_embeddings[cond_indices]
                    if len(self._item_vectors) > 0:
                        retrieval_item_vectors = self._item_vectors[cond_indices]
                else:
                    retrieval_cards = []
                    retrieval_embeddings = np.zeros((0, self._memory_embeddings.shape[1]), dtype=np.float32) if len(self._memory_embeddings) > 0 else self._memory_embeddings
                    retrieval_item_vectors = np.zeros((0, self._rl_cfg["item_dim"]), dtype=np.float32)

            # Query transformation for better retrieval matching
            retrieval_queries = None
            if self.enable_query_transform:
                retrieval_queries = self._transform_query_for_retrieval(query)

            # Retrieve memories
            if self.mode == "nopersonal":
                candidates, cand_item_vecs, base_scores, chosen_indices, probs = retrieve_no_policy(
                    user_id=user_id,
                    query=query,
                    embed_model=self._embed_model,
                    reranker=self._reranker,
                    memory_cards=retrieval_cards,
                    memory_embeddings=retrieval_embeddings,
                    topk_dense=self._rl_cfg["dense_topk"],
                    topk_rerank=self._rl_cfg["rerank_topk"],
                    only_own_memories=self.only_own_memories,
                    queries=retrieval_queries,
                    dynamic_topk=self._rl_cfg["dynamic_topk"],
                    dynamic_min_k=self._rl_cfg["dynamic_min_k"],
                    dynamic_max_k=self._rl_cfg["dynamic_max_k"],
                    dynamic_score_ratio=self._rl_cfg["dynamic_score_ratio"],
                )
            else:
                beta_long = self._rl_cfg["beta_long"]
                beta_short = self._rl_cfg["beta_short"]
                candidates, cand_item_vecs, base_scores, chosen_indices, probs = retrieve_with_policy(
                    user_id=user_id,
                    query=query,
                    embed_model=self._embed_model,
                    reranker=self._reranker,
                    memory_cards=retrieval_cards,
                    memory_embeddings=retrieval_embeddings,
                    user_store=self._user_store,
                    item_vectors=retrieval_item_vectors,
                    topk_dense=self._rl_cfg["dense_topk"],
                    topk_rerank=self._rl_cfg["rerank_topk"],
                    beta_long=beta_long,
                    beta_short=beta_short,
                    tau=self._rl_cfg["tau"],
                    only_own_memories=self.only_own_memories,
                    sample=not self.eval_mode,
                    queries=retrieval_queries,
                )

            # Get selected memories
            memories_t = [candidates[int(i)] for i in chosen_indices] if chosen_indices else []
            memory_notes = [m.note_text for m in memories_t]

            # Apply preference rewrite if enabled
            if self.enable_preference_rewrite and memory_notes:
                memory_notes = self._rewrite_preferences(memory_notes, query)

            # Debug: show retrieval info
            if memories_t or global_notes:
                print(f"[DEBUG-RETRIEVAL] User={user_id}, Query={query[:50]}...")
                print(f"[DEBUG-RETRIEVAL]   Global={len(global_notes)}, Candidates={len(candidates)}, Retrieved={len(memories_t)}")
                for i, m in enumerate(memories_t[:3]):  # Show top 3
                    score = probs[chosen_indices[i]] if i < len(chosen_indices) and chosen_indices[i] < len(probs) else 0
                    print(f"[DEBUG-RETRIEVAL]   [{i+1}] score={score:.3f}: {m.note_text[:80]}...")

        # Combine all notes for prompt (global + retrieved)
        # For chat(), we combine all notes; chat_prepare() handles them separately
        if self.mode != "vanilla":
            all_memory_notes = (global_notes if global_notes else []) + memory_notes
        else:
            all_memory_notes = memory_notes

        # Build prompt and count tokens
        prompt_tokens = self._count_tokens(query)
        for turn in session.history:
            prompt_tokens += self._count_tokens(turn.text)
        for note in all_memory_notes:
            prompt_tokens += self._count_tokens(note)

        # Generate answer (with best-of-N if enabled)
        if self.best_of_n > 1:
            # Generate N responses and pick the best one
            candidates_responses = []
            for i in range(self.best_of_n):
                resp = self._chat_model.answer(
                    history=session.history,
                    memory_notes=all_memory_notes,
                    max_new_tokens=self._rl_cfg["max_new_tokens"],
                    temperature=0.8,  # Slightly higher temp for diversity
                )
                score = self._score_response(resp)
                candidates_responses.append((resp, score))

            # Sort by score (descending) and pick best
            candidates_responses.sort(key=lambda x: x[1], reverse=True)
            answer_t = candidates_responses[0][0]
            best_score = candidates_responses[0][1]

            if len(candidates_responses) > 1:
                print(f"[BEST-OF-{self.best_of_n}] Scores: {[f'{s:.2f}' for _, s in candidates_responses]}, picked score={best_score:.2f}")
        else:
            answer_t = self._chat_model.answer(
                history=session.history,
                memory_notes=all_memory_notes,
                max_new_tokens=self._rl_cfg["max_new_tokens"],
            )

        completion_tokens = self._count_tokens(answer_t)
        
        # Add assistant turn to history
        assist_turn = self._build_chat_turn(user_id, answer_t, "assistant", ctx.turn_counter)
        session.history.append(assist_turn)
        
        # Update session state for next turn
        session.last_query = query
        session.last_answer = answer_t
        session.last_memories = memories_t
        session.last_query_embedding = e_q_t
        session.last_candidate_item_vectors = cand_item_vecs
        session.last_policy_probs = probs
        session.last_chosen_indices = list(chosen_indices) if len(chosen_indices) > 0 else []
        
        ctx.turn_counter += 1
        
        # Build debug info
        debug = DebugInfo(
            selected_memory_ids=[m.card_id for m in memories_t],
            selected_memory_notes=[m.note_text for m in memories_t],
            selected_memory_scores=[float(probs[i]) if i < len(probs) else 0.0 for i in chosen_indices] if len(chosen_indices) > 0 else [],
            user_vector_before=z_long_before + z_short_before,  # Concatenated for simplicity
            user_vector_after=user_state.z_long.tolist() + user_state.z_short.tolist(),
            extracted_preferences=extracted_prefs,
            extra={
                "num_candidates": len(candidates),
                "num_total_memories": len(self._memory_cards),
                "z_long_norm": float(np.linalg.norm(user_state.z_long)),
                "z_short_norm": float(np.linalg.norm(user_state.z_short)),
            }
        )
        
        # Build usage stats
        usage = UsageStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            model=self._llm_name,
        )
        
        return AssistantResponse(
            answer=answer_t,
            usage=usage,
            debug=debug,
        )

    def chat_prepare(self, user_id: str, query: str, skip_extraction: bool = False, skip_auto_reward: bool = False) -> dict:
        """
        Prepare for chat without calling the LLM.

        This does all the preparation work (embedding, memory retrieval, etc.)
        and returns the messages to send to the LLM along with context needed
        for post-processing.

        Used for batch processing where messages are collected first, then
        sent in batch to vLLM for concurrent processing.

        Args:
            user_id: Unique identifier for the user.
            query: Current user query/message.

        Returns:
            Dict containing:
                - messages: List of messages to send to LLM
                - context: Dict with all state needed for chat_complete()
        """
        ctx = self._get_or_create_session(user_id)
        session = ctx.session_state
        user_state = self._user_store.get_state(user_id)

        # Record user vector before for debug
        z_long_before = user_state.z_long.copy().tolist()
        z_short_before = user_state.z_short.copy().tolist()

        # Add user turn to history
        user_turn = self._build_chat_turn(user_id, query, "user", ctx.turn_counter)
        session.history.append(user_turn)

        # Vanilla mode: pure LLM without any memory or preference extraction
        if self.mode == "vanilla":
            e_q_t = np.zeros(4096, dtype=np.float32)
            extracted_prefs = []
            candidates = []
            cand_item_vecs = np.array([])
            base_scores = np.array([])
            chosen_indices = []
            probs = np.array([])
            memories_t = []
            memory_notes = []
        else:
            # Compute query embedding
            embed_result = self._embed_model.encode([query], normalize=True, return_tensor=False)
            if embed_result is None or len(embed_result) == 0:
                raise RuntimeError(f"Embedding model returned empty result for query: {query[:100]}")
            e_q_t = np.array(embed_result[0])

            # Store pending RL update info from last turn
            if session.last_query is not None and self.enable_rl_updates:
                ctx.pending_rl_update = {
                    "last_query": session.last_query,
                    "last_answer": session.last_answer,
                    "last_memories": session.last_memories,
                    "last_query_embedding": session.last_query_embedding,
                    "current_query_embedding": e_q_t,
                    "last_candidate_item_vectors": session.last_candidate_item_vectors,
                    "last_policy_probs": session.last_policy_probs,
                    "last_chosen_indices": session.last_chosen_indices,
                }

                # Auto-compute reward via LLM judge if enabled
                # skip_auto_reward=True when batch framework handles rewards externally
                if self._llm_reward_client is not None and not skip_auto_reward:
                    import asyncio
                    try:
                        reward, gating = asyncio.run(eval_step_llm(
                            q_t=session.last_query,
                            answer_t=session.last_answer,
                            q_t1=query,
                            memories_t=session.last_memories or [],
                            client=self._llm_reward_client,
                        ))
                        if gating > 0.0:
                            self.apply_feedback(Feedback(
                                user_id=user_id,
                                turn_id=ctx.turn_counter - 1,
                                reward=reward,
                                gating=gating,
                            ))
                    except Exception as e:
                        print(f"[LLM-Reward] Judge call failed, skipping update: {e}")

            # Extract preferences from conversation
            extracted_prefs = []
            if self.enable_preference_extraction and not skip_extraction:
                prefs = self._extractor.extract_turn(session.history)
                if prefs.preferences:
                    print(f"[DEBUG] Extracted {len(prefs.preferences)} prefs from history (len={len(session.history)})")
                extracted_prefs = self._add_preferences_as_memory(
                    prefs, query, user_id, ctx.turn_counter
                )
                if extracted_prefs:
                    print(f"[DEBUG] Added {len(extracted_prefs)} to memory. Total cards: {len(self._memory_cards)}")

            # Separate global preferences (bypass retrieval) from conditional ones
            global_notes = []
            retrieval_cards = self._memory_cards
            retrieval_embeddings = self._memory_embeddings
            retrieval_item_vectors = self._item_vectors
            if self.enable_global_preferences:
                global_cards = [c for c in self._memory_cards if c.is_global and c.user_id == user_id]
                global_notes = [c.note_text for c in global_cards[:10]]  # Cap at 10
                cond_indices = [i for i, c in enumerate(self._memory_cards) if not c.is_global]
                if cond_indices:
                    retrieval_cards = [self._memory_cards[i] for i in cond_indices]
                    retrieval_embeddings = self._memory_embeddings[cond_indices]
                    if len(self._item_vectors) > 0:
                        retrieval_item_vectors = self._item_vectors[cond_indices]
                else:
                    retrieval_cards = []
                    retrieval_embeddings = np.zeros((0, self._memory_embeddings.shape[1]), dtype=np.float32) if len(self._memory_embeddings) > 0 else self._memory_embeddings
                    retrieval_item_vectors = np.zeros((0, self._rl_cfg["item_dim"]), dtype=np.float32)

            # Query transformation for better retrieval matching
            retrieval_queries = None
            if self.enable_query_transform:
                retrieval_queries = self._transform_query_for_retrieval(query)

            # Retrieve memories
            if self.mode == "nopersonal":
                candidates, cand_item_vecs, base_scores, chosen_indices, probs = retrieve_no_policy(
                    user_id=user_id,
                    query=query,
                    embed_model=self._embed_model,
                    reranker=self._reranker,
                    memory_cards=retrieval_cards,
                    memory_embeddings=retrieval_embeddings,
                    topk_dense=self._rl_cfg["dense_topk"],
                    topk_rerank=self._rl_cfg["rerank_topk"],
                    only_own_memories=self.only_own_memories,
                    queries=retrieval_queries,
                    dynamic_topk=self._rl_cfg["dynamic_topk"],
                    dynamic_min_k=self._rl_cfg["dynamic_min_k"],
                    dynamic_max_k=self._rl_cfg["dynamic_max_k"],
                    dynamic_score_ratio=self._rl_cfg["dynamic_score_ratio"],
                )
            else:
                beta_long = self._rl_cfg["beta_long"]
                beta_short = self._rl_cfg["beta_short"]
                candidates, cand_item_vecs, base_scores, chosen_indices, probs = retrieve_with_policy(
                    user_id=user_id,
                    query=query,
                    embed_model=self._embed_model,
                    reranker=self._reranker,
                    memory_cards=retrieval_cards,
                    memory_embeddings=retrieval_embeddings,
                    user_store=self._user_store,
                    item_vectors=retrieval_item_vectors,
                    topk_dense=self._rl_cfg["dense_topk"],
                    topk_rerank=self._rl_cfg["rerank_topk"],
                    beta_long=beta_long,
                    beta_short=beta_short,
                    tau=self._rl_cfg["tau"],
                    only_own_memories=self.only_own_memories,
                    sample=not self.eval_mode,
                    queries=retrieval_queries,
                )

            memories_t = [candidates[int(i)] for i in chosen_indices] if chosen_indices else []
            memory_notes = [m.note_text for m in memories_t]

            # Apply preference rewrite if enabled
            if self.enable_preference_rewrite and memory_notes:
                memory_notes = self._rewrite_preferences(memory_notes, query)

            if memories_t or global_notes:
                print(f"[DEBUG-RETRIEVAL] User={user_id}, Query={query[:50]}...")
                print(f"[DEBUG-RETRIEVAL]   Global={len(global_notes)}, Candidates={len(candidates)}, Retrieved={len(memories_t)}")
                for i, m in enumerate(memories_t[:3]):
                    score = probs[chosen_indices[i]] if i < len(chosen_indices) and chosen_indices[i] < len(probs) else 0
                    print(f"[DEBUG-RETRIEVAL]   [{i+1}] score={score:.3f}: {m.note_text[:80]}...")

        # Build prompt token count
        prompt_tokens = self._count_tokens(query)
        for turn in session.history:
            prompt_tokens += self._count_tokens(turn.text)
        all_notes = memory_notes + (global_notes if self.mode != "vanilla" else [])
        for note in all_notes:
            prompt_tokens += self._count_tokens(note)

        # Build messages for LLM (pass global_notes separately for distinct prompt sections)
        effective_global = global_notes if (self.enable_global_preferences and self.mode != "vanilla") else None
        messages = self._chat_model.build_messages(
            history=session.history,
            memory_notes=memory_notes,
            max_new_tokens=self._rl_cfg["max_new_tokens"],
            global_notes=effective_global,
        )

        # Return messages and context for chat_complete
        return {
            "messages": messages,
            "context": {
                "user_id": user_id,
                "query": query,
                "ctx": ctx,
                "session": session,
                "user_state": user_state,
                "z_long_before": z_long_before,
                "z_short_before": z_short_before,
                "e_q_t": e_q_t,
                "extracted_prefs": extracted_prefs,
                "candidates": candidates,
                "cand_item_vecs": cand_item_vecs,
                "base_scores": base_scores,
                "chosen_indices": chosen_indices,
                "probs": probs,
                "memories_t": memories_t,
                "memory_notes": memory_notes,
                "prompt_tokens": prompt_tokens,
            }
        }

    def chat_complete(self, answer_t: str, context: dict) -> AssistantResponse:
        """
        Complete chat with LLM response.

        This takes the LLM response and context from chat_prepare(), and
        does all post-processing (add to history, debug info, etc.).

        Args:
            answer_t: The LLM response text.
            context: Context dict from chat_prepare().

        Returns:
            AssistantResponse containing the answer, usage stats, and debug info.
        """
        # Unpack context
        user_id = context["user_id"]
        query = context["query"]
        ctx = context["ctx"]
        session = context["session"]
        user_state = context["user_state"]
        z_long_before = context["z_long_before"]
        z_short_before = context["z_short_before"]
        e_q_t = context["e_q_t"]
        extracted_prefs = context["extracted_prefs"]
        candidates = context["candidates"]
        cand_item_vecs = context["cand_item_vecs"]
        chosen_indices = context["chosen_indices"]
        probs = context["probs"]
        memories_t = context["memories_t"]
        memory_notes = context["memory_notes"]
        prompt_tokens = context["prompt_tokens"]

        completion_tokens = self._count_tokens(answer_t)

        # Add assistant turn to history
        assist_turn = self._build_chat_turn(user_id, answer_t, "assistant", ctx.turn_counter)
        session.history.append(assist_turn)

        # Update session state for next turn
        session.last_query = query
        session.last_answer = answer_t
        session.last_memories = memories_t
        session.last_query_embedding = e_q_t
        session.last_candidate_item_vectors = cand_item_vecs
        session.last_policy_probs = probs
        session.last_chosen_indices = list(chosen_indices) if len(chosen_indices) > 0 else []

        ctx.turn_counter += 1

        # Build debug info
        debug = DebugInfo(
            selected_memory_ids=[m.card_id for m in memories_t],
            selected_memory_notes=[m.note_text for m in memories_t],
            selected_memory_scores=[float(probs[i]) if i < len(probs) else 0.0 for i in chosen_indices] if len(chosen_indices) > 0 else [],
            user_vector_before=z_long_before + z_short_before,
            user_vector_after=user_state.z_long.tolist() + user_state.z_short.tolist(),
            extracted_preferences=extracted_prefs,
            extra={
                "num_candidates": len(candidates),
                "num_total_memories": len(self._memory_cards),
                "z_long_norm": float(np.linalg.norm(user_state.z_long)),
                "z_short_norm": float(np.linalg.norm(user_state.z_short)),
            }
        )

        # Build usage stats
        usage = UsageStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            model=self._llm_name,
        )

        return AssistantResponse(
            answer=answer_t,
            usage=usage,
            debug=debug,
        )

    def apply_extracted_preferences(self, user_id: str, pref_dict: dict) -> list:
        """Apply pre-computed extraction results (from batch extraction) to memory."""
        prefs = PreferenceList.model_validate(pref_dict)
        if not prefs.preferences:
            return []
        ctx = self._get_or_create_session(user_id)
        query = ctx.session_state.history[-1].text if ctx.session_state.history else ""
        extracted = self._add_preferences_as_memory(prefs, query, user_id, ctx.turn_counter)
        if extracted:
            print(f"[DEBUG] Batch-added {len(extracted)} to memory. Total cards: {len(self._memory_cards)}")
        return extracted

    def get_last_user_query(self, user_id: str) -> str:
        """Get the last user message text for this user's session."""
        ctx = self._sessions.get(user_id)
        if ctx and ctx.session_state.history:
            for t in reversed(ctx.session_state.history):
                if t.role == "user":
                    return t.text
        return ""

    def reset_session(self, user_id: str) -> None:
        """
        Reset session for a user (new chat window).

        This clears:
        - Session conversation history
        - Short-term user vector (z_short)
        - Pending RL update info

        This preserves:
        - Long-term user vector (z_long)
        - User's memory cards (may be consolidated if enabled)

        Args:
            user_id: The user whose session to reset.
        """
        # Consolidate preferences at session end (before clearing session)
        if self.enable_preference_consolidation:
            self.consolidate_user_preferences(user_id)

        # Clear session context
        if user_id in self._sessions:
            del self._sessions[user_id]
        
        # Create fresh session
        self._sessions[user_id] = _SessionContext(
            session_state=OnlineSessionState(user_id=user_id),
            turn_counter=0,
        )
        
        # Reset short-term vector but keep long-term
        user_state = self._user_store.get_state(user_id)
        user_state.z_short = np.zeros(self._rl_cfg["item_dim"], dtype=np.float32)
        self._user_store.save_state(user_state)
    
    def reset_user(self, user_id: str) -> None:
        """
        Completely reset a user (new "life").
        
        This clears:
        - Long-term user vector (z_long)
        - Short-term user vector (z_short)
        - User's memory cards
        - Session history
        - All cached state
        
        Args:
            user_id: The user to reset.
        """
        # Clear session
        if user_id in self._sessions:
            del self._sessions[user_id]
        
        # Reset user state vectors
        user_state = self._user_store.get_state(user_id)
        user_state.z_long = self._user_store.global_init_z.copy()
        user_state.z_short = np.zeros(self._rl_cfg["item_dim"], dtype=np.float32)
        user_state.reward_ma = 0.0
        self._user_store.save_state(user_state)
        
        # Find indices to KEEP (cards NOT belonging to this user)
        # Must do this BEFORE modifying _memory_cards
        keep_indices = [
            i for i, card in enumerate(self._memory_cards)
            if card.user_id != user_id
        ]
        
        # Filter memory cards
        self._memory_cards = [self._memory_cards[i] for i in keep_indices]
        
        # Filter embeddings and item vectors to match
        if len(keep_indices) > 0 and len(self._memory_embeddings) > 0:
            self._memory_embeddings = self._memory_embeddings[keep_indices]
            self._item_vectors = self._item_vectors[keep_indices]
        else:
            # No cards left or no embeddings
            embed_dim = self._memory_embeddings.shape[1] if len(self._memory_embeddings) > 0 else 4096
            self._memory_embeddings = np.zeros((0, embed_dim), dtype=np.float32)
            self._item_vectors = np.zeros((0, self._rl_cfg["item_dim"]), dtype=np.float32)
    
    def apply_feedback(self, feedback: Feedback) -> None:
        """
        Apply feedback from user simulator or judge.
        
        This performs the REINFORCE update to user vectors based on
        the reward signal from the previous turn.
        
        Args:
            feedback: Feedback object containing reward, gating, and metadata.
        
        Notes:
            - Should be called AFTER chat() but BEFORE the next chat() call
            - Uses the stored context from the previous turn
            - If enable_rl_updates is False, this is a no-op (logging only)
            - If mode is "nopersonal", this is a no-op (baseline comparison)
        """
        if not self.enable_rl_updates:
            return

        # In "nopersonal" or "vanilla" mode, skip RL updates entirely (baseline)
        if self.mode in ("nopersonal", "vanilla"):
            return

        user_id = feedback.user_id
        ctx = self._sessions.get(user_id)

        if ctx is None or ctx.pending_rl_update is None:
            return
        
        pending = ctx.pending_rl_update
        user_state = self._user_store.get_state(user_id)
        
        # Check if we have the necessary data for RL update
        if (pending.get("last_candidate_item_vectors") is not None and
            pending.get("last_policy_probs") is not None and
            pending.get("last_chosen_indices") is not None and
            len(pending["last_chosen_indices"]) > 0):

            # Extract chosen vectors
            chosen_indices = pending["last_chosen_indices"]
            candidate_vectors = pending["last_candidate_item_vectors"]

            if len(candidate_vectors) > 0:
                print(f"[DEBUG-REINFORCE] User={user_id} reward={feedback.reward:.2f} "
                      f"n_candidates={len(candidate_vectors)} chosen={chosen_indices} "
                      f"probs_shape={pending['last_policy_probs'].shape if hasattr(pending['last_policy_probs'], 'shape') else 'N/A'}")
                # REINFORCE expects:
                # - item_vectors: ALL candidate vectors [K, k]
                # - chosen_indices: indices into those candidates
                # - policy_probs: probabilities over all K candidates [K]
                updated = reinforce_update_user_state(
                    user_state=user_state,
                    item_vectors=candidate_vectors,  # All candidates, not just chosen
                    chosen_indices=chosen_indices,   # Original indices into candidates
                    policy_probs=pending["last_policy_probs"],
                    reward_hat=feedback.reward,
                    gating=feedback.gating,
                    tau=self._rl_cfg["tau"],
                    eta_long=self._rl_cfg["eta_long"],
                    eta_short=self._rl_cfg["eta_short"],
                    ema_alpha=self._rl_cfg["ema_alpha"],
                    short_decay=self._rl_cfg["short_decay"],
                )
                
                print(f"[DEBUG-REINFORCE] updated={updated} z_long_norm={np.linalg.norm(user_state.z_long):.15e}")
                if updated:
                    self._user_store.save_state(user_state)
        
        # Clear pending update
        ctx.pending_rl_update = None
    
    def get_user_state_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get a summary of the user's current state (for debugging/analysis).
        
        Args:
            user_id: The user to query.
        
        Returns:
            Dictionary with user state information.
        """
        user_state = self._user_store.get_state(user_id)
        ctx = self._sessions.get(user_id)
        
        user_memory_count = sum(
            1 for card in self._memory_cards if card.user_id == user_id
        )
        
        return {
            "user_id": user_id,
            "z_long_norm": float(np.linalg.norm(user_state.z_long)),
            "z_short_norm": float(np.linalg.norm(user_state.z_short)),
            "reward_ma": user_state.reward_ma,
            "session_history_length": len(ctx.session_state.history) if ctx else 0,
            "turn_counter": ctx.turn_counter if ctx else 0,
            "user_memory_count": user_memory_count,
            "total_memory_count": len(self._memory_cards),
        }
    
    def persist(self) -> None:
        """
        Persist all state to disk.
        
        Call this at the end of an evaluation run to save:
        - User state vectors
        - Memory cards
        """
        # Save user store
        self._user_store.persist()
        
        # Save memory cards
        with open(self._memory_cards_path, "w", encoding="utf-8") as f:
            for card in self._memory_cards:
                f.write(card.model_dump_json() + "\n")
        
        # Save embeddings
        np.save(self._memory_embeddings_path, self._memory_embeddings)
        
        # Save item projection with updated vectors
        if self._projection is not None:
            np.savez(
                self._item_projection_path,
                P=self._projection.P,
                mean=self._projection.mean,
                V=self._item_vectors,
            )
        
        print("[PersonalizedLLM] State persisted to disk.")


# =============================================================================
# Convenience Factory
# =============================================================================

def create_personalized_llm(
    config_path: Optional[str] = None,
    **kwargs
) -> PersonalizedLLM:
    """
    Factory function to create a PersonalizedLLM instance.
    
    Args:
        config_path: Optional path to configuration file.
        **kwargs: Additional arguments passed to PersonalizedLLM constructor.
    
    Returns:
        Configured PersonalizedLLM instance.
    """
    return PersonalizedLLM(config_path=config_path, **kwargs)

