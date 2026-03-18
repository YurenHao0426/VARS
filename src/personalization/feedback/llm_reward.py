"""
LLM-as-Judge reward model using OpenAI GPT-5-nano (async for parallelism).

Replaces keyword-based heuristic reward with structured LLM judgement.
Judge receives only (q_t, a_t, q_{t+1}) — no oracle preference cards, no history.

Label taxonomy → scalar reward mapping:
    neg_constraint_restate  → -1.0
    neg_correction          → -0.8
    neg_confusion           → -0.6
    pos_praise              → +0.8
    pos_progress            → +0.1
    neutral                 →  0.0
    topic_shift             →  0.0 (update skipped)

Confidence gating: if confidence < tau_c, reward is set to 0 and update is skipped.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from openai import AsyncOpenAI, RateLimitError, APITimeoutError, APIConnectionError

from personalization.feedback.schemas import TurnSample


# --- Label → Reward Mapping ---

REWARD_MAP: Dict[str, float] = {
    "neg_constraint_restate": -1.0,
    "neg_correction": -0.8,
    "neg_confusion": -0.6,
    "pos_praise": +0.8,
    "pos_progress": +0.1,
    "neutral": 0.0,
    "topic_shift": 0.0,
}

VALID_LABELS = set(REWARD_MAP.keys())


# --- Configuration ---

@dataclass
class LLMRewardConfig:
    model: str = "gpt-5-nano"
    api_key: Optional[str] = None          # Falls back to OPENAI_API_KEY env var
    base_url: Optional[str] = None         # For custom endpoints
    max_concurrent: int = 32               # Semaphore limit for parallel requests
    max_retries: int = 3
    retry_base_delay: float = 1.0          # Exponential backoff base (seconds)
    timeout: float = 60.0                  # Per-request timeout (reasoning models are slower)
    max_completion_tokens: int = 2048      # Must be high — reasoning models use internal tokens
    confidence_threshold: float = 0.6     # tau_c: skip update if confidence < this
    enable_cache: bool = True              # Cache by hash of (q_t, a_t, q_{t+1})


# --- Prompt ---

JUDGE_SYSTEM_PROMPT = """\
You are a feedback classifier. Given a user query (q_t), the assistant's response (a_t), \
and the user's next message (q_{t+1}), classify the user's follow-up into exactly one label.

Labels (mutually exclusive):
- neg_constraint_restate: User reasserts constraints/preferences as correction (e.g., "as I said…", "remember…", "按我说的…").
- neg_correction: User indicates the content is wrong or the assistant failed to answer.
- neg_confusion: User indicates confusion or requests re-explanation.
- pos_praise: Explicit praise or satisfaction with the response.
- pos_progress: Constructive continuation (examples, extensions, what-if, next steps) without complaint.
- neutral: Ambiguous or minimal feedback, not clearly positive or negative.
- topic_shift: User switches to a new unrelated task/topic.

Output a JSON object with fields: label, confidence (0-1), rationale (one short sentence)."""

JUDGE_USER_TEMPLATE = """\
q_t: {query_t}

a_t: {answer_t}

q_{{t+1}}: {query_t1}"""


# --- Result Dataclass ---

@dataclass
class RewardResult:
    label: str
    confidence: float
    rationale: str
    reward: float
    should_update: bool  # False if gated by confidence or topic_shift


# --- Async Client ---

class LLMRewardClient:
    """Async OpenAI client for LLM-as-judge reward estimation."""

    def __init__(self, config: Optional[LLMRewardConfig] = None):
        self.config = config or LLMRewardConfig()
        self._client = AsyncOpenAI(
            api_key=self.config.api_key or os.getenv("OPENAI_API_KEY"),
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        self._cache: Dict[str, RewardResult] = {}

    def _cache_key(self, query_t: str, answer_t: str, query_t1: str) -> str:
        """Deterministic hash of the judge input triple."""
        content = f"{query_t}\x00{answer_t}\x00{query_t1}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    async def _call_with_retry(self, messages: List[dict]) -> str:
        """Single LLM call with exponential backoff retry."""
        for attempt in range(self.config.max_retries):
            try:
                async with self._semaphore:
                    response = await self._client.chat.completions.create(
                        model=self.config.model,
                        messages=messages,
                        max_completion_tokens=self.config.max_completion_tokens,
                        response_format={"type": "json_object"},
                    )
                content = response.choices[0].message.content
                if content:
                    return content.strip()
                # Reasoning model may exhaust tokens on thinking — retry
                if response.choices[0].finish_reason == "length":
                    continue
                return ""
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                if attempt == self.config.max_retries - 1:
                    raise
                delay = self.config.retry_base_delay * (2 ** attempt)
                await asyncio.sleep(delay)
        return ""

    def _build_messages(self, sample: TurnSample) -> List[dict]:
        """Construct the judge prompt from (q_t, a_t, q_{t+1}) only."""
        user_content = JUDGE_USER_TEMPLATE.format(
            query_t=sample.query_t,
            answer_t=sample.answer_t,
            query_t1=sample.query_t1,
        )
        return [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _parse_result(self, raw: str) -> RewardResult:
        """Parse structured JSON output into RewardResult."""
        try:
            parsed = json.loads(raw)
            label = parsed["label"]
            confidence = float(parsed["confidence"])
            rationale = parsed.get("rationale", "")

            if label not in VALID_LABELS:
                label = "neutral"
                confidence = 0.0

            reward = REWARD_MAP[label]

            # Confidence gating and topic_shift skip
            should_update = (
                confidence >= self.config.confidence_threshold
                and label != "topic_shift"
            )
            if not should_update:
                reward = 0.0

            return RewardResult(
                label=label,
                confidence=confidence,
                rationale=rationale,
                reward=reward,
                should_update=should_update,
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return RewardResult(
                label="neutral",
                confidence=0.0,
                rationale="parse_failure",
                reward=0.0,
                should_update=False,
            )

    async def judge(self, sample: TurnSample) -> RewardResult:
        """Judge a single turn (async). Returns RewardResult with gating applied."""
        # Cache lookup
        if self.config.enable_cache:
            key = self._cache_key(sample.query_t, sample.answer_t, sample.query_t1)
            if key in self._cache:
                return self._cache[key]

        messages = self._build_messages(sample)
        raw = await self._call_with_retry(messages)
        result = self._parse_result(raw)

        # Cache store
        if self.config.enable_cache:
            self._cache[key] = result

        return result

    async def judge_batch(self, samples: List[TurnSample]) -> List[RewardResult]:
        """Judge a batch of turns in parallel. Returns list of RewardResult."""
        tasks = [self.judge(s) for s in samples]
        return await asyncio.gather(*tasks)

    async def close(self):
        """Close the underlying HTTP client."""
        await self._client.close()


# --- Synchronous Wrappers ---

def estimate_reward_llm(
    sample: TurnSample,
    config: Optional[LLMRewardConfig] = None,
) -> Tuple[float, bool]:
    """
    Synchronous single-sample reward estimation.
    Returns (reward, should_update).
    """
    client = LLMRewardClient(config)
    try:
        result = asyncio.run(client.judge(sample))
        return result.reward, result.should_update
    finally:
        asyncio.run(client.close())


def estimate_rewards_batch(
    samples: List[TurnSample],
    config: Optional[LLMRewardConfig] = None,
) -> List[Tuple[float, bool]]:
    """
    Synchronous batch reward estimation (runs async internally).
    Returns list of (reward, should_update) tuples.
    """
    client = LLMRewardClient(config)
    try:
        results = asyncio.run(client.judge_batch(samples))
        return [(r.reward, r.should_update) for r in results]
    finally:
        asyncio.run(client.close())
