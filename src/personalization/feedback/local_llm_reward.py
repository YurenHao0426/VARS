"""
Local LLM reward model using vLLM server for batch inference.

Drop-in replacement for LLMRewardClient when you want to use a local model
(e.g., Llama-3.1-8B-Instruct) instead of OpenAI API.

Uses BatchVLLMClient for efficient concurrent requests - vLLM's continuous
batching will process them together for high throughput.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import aiohttp

from personalization.feedback.schemas import TurnSample
from personalization.feedback.llm_reward import (
    REWARD_MAP,
    VALID_LABELS,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
    RewardResult,
)


@dataclass
class LocalLLMRewardConfig:
    """Configuration for local LLM reward model."""
    vllm_url: str = "http://localhost:8005/v1"  # vLLM server URL
    model_name: Optional[str] = None  # Auto-discovered if None
    max_tokens: int = 256
    temperature: float = 0.1
    max_concurrent: int = 100  # High concurrency for vLLM batching
    timeout: Optional[int] = 60  # Per-request timeout in seconds
    confidence_threshold: float = 0.6  # tau_c: skip update if confidence < this
    enable_cache: bool = True  # Cache by hash of (q_t, a_t, q_{t+1})


class LocalLLMRewardClient:
    """
    Local LLM reward client using vLLM server.

    Designed for batch processing - uses async HTTP requests that vLLM
    batches together via continuous batching for high throughput.
    """

    def __init__(self, config: Optional[LocalLLMRewardConfig] = None):
        self.config = config or LocalLLMRewardConfig()
        self._model_name = self.config.model_name
        self._cache: Dict[str, RewardResult] = {}

        # Discover model name if not provided
        if self._model_name is None:
            self._discover_model_sync()

    def _discover_model_sync(self):
        """Synchronously discover model name from vLLM server."""
        import requests
        try:
            response = requests.get(
                f"{self.config.vllm_url}/models",
                timeout=10
            )
            response.raise_for_status()
            models = response.json()
            if models.get("data") and len(models["data"]) > 0:
                self._model_name = models["data"][0]["id"]
            else:
                self._model_name = "default"
        except Exception as e:
            print(f"[LocalLLMReward] Warning: Could not discover model ({e})")
            self._model_name = "default"

    def _cache_key(self, query_t: str, answer_t: str, query_t1: str) -> str:
        """Deterministic hash of the judge input triple."""
        content = f"{query_t}\x00{answer_t}\x00{query_t1}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _build_messages(self, sample: TurnSample) -> List[dict]:
        """Construct the judge prompt from (q_t, a_t, q_{t+1})."""
        user_content = JUDGE_USER_TEMPLATE.format(
            query_t=sample.query_t,
            answer_t=sample.answer_t,
            query_t1=sample.query_t1,
        )
        return [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _parse_result(self, raw: Optional[str]) -> RewardResult:
        """Parse structured JSON output into RewardResult."""
        if raw is None:
            return RewardResult(
                label="neutral",
                confidence=0.0,
                rationale="request_failed",
                reward=0.0,
                should_update=False,
            )

        try:
            # Handle markdown code blocks
            text = raw.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(
                    lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
                )

            parsed = json.loads(text)
            label = parsed.get("label", "neutral")
            confidence = float(parsed.get("confidence", 0.0))
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
            # Try to extract JSON from text
            import re
            match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                    label = parsed.get("label", "neutral")
                    confidence = float(parsed.get("confidence", 0.0))
                    rationale = parsed.get("rationale", "")

                    if label not in VALID_LABELS:
                        label = "neutral"
                        confidence = 0.0

                    reward = REWARD_MAP[label]
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
                except:
                    pass

            return RewardResult(
                label="neutral",
                confidence=0.0,
                rationale="parse_failure",
                reward=0.0,
                should_update=False,
            )

    async def _single_request(
        self,
        session: aiohttp.ClientSession,
        messages: List[dict],
        idx: int,
    ) -> tuple:
        """Make a single async request to vLLM server."""
        payload = {
            "model": self._model_name,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "response_format": {"type": "json_object"},
        }

        for attempt in range(3):
            try:
                timeout_config = (
                    aiohttp.ClientTimeout(total=self.config.timeout)
                    if self.config.timeout else None
                )
                async with session.post(
                    f"{self.config.vllm_url}/chat/completions",
                    json=payload,
                    timeout=timeout_config,
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result["choices"][0]["message"]["content"]
                        return (idx, content, None)
                    elif response.status == 429:
                        # Rate limit - wait and retry
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        error_text = await response.text()
                        return (idx, None, f"HTTP {response.status}: {error_text[:200]}")
            except asyncio.TimeoutError:
                if attempt < 2:
                    continue
                return (idx, None, "Timeout")
            except Exception as e:
                return (idx, None, str(e))

        return (idx, None, "Max retries exceeded")

    async def judge_batch_async(
        self,
        samples: List[TurnSample],
        show_progress: bool = False,
    ) -> List[RewardResult]:
        """
        Judge a batch of turns using concurrent vLLM requests.

        vLLM's continuous batching will process these together for
        high throughput.
        """
        n_samples = len(samples)
        results = [None] * n_samples

        # Check cache and build request list
        to_request = []  # (original_idx, messages)
        for i, sample in enumerate(samples):
            if self.config.enable_cache:
                key = self._cache_key(sample.query_t, sample.answer_t, sample.query_t1)
                if key in self._cache:
                    results[i] = self._cache[key]
                    continue

            messages = self._build_messages(sample)
            to_request.append((i, messages))

        if not to_request:
            return results

        # Make concurrent requests
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def limited_request(session, messages, idx):
            async with semaphore:
                return await self._single_request(session, messages, idx)

        connector = aiohttp.TCPConnector(limit=self.config.max_concurrent)
        headers = {"Content-Type": "application/json"}

        async with aiohttp.ClientSession(
            connector=connector, headers=headers
        ) as session:
            tasks = [
                limited_request(session, messages, orig_idx)
                for orig_idx, messages in to_request
            ]

            completed = 0
            for coro in asyncio.as_completed(tasks):
                orig_idx, content, error = await coro
                completed += 1

                if error:
                    print(f"[LocalLLMReward] Request {orig_idx} failed: {error}")

                result = self._parse_result(content)
                results[orig_idx] = result

                # Cache the result
                if self.config.enable_cache:
                    sample = samples[orig_idx]
                    key = self._cache_key(
                        sample.query_t, sample.answer_t, sample.query_t1
                    )
                    self._cache[key] = result

                if show_progress and completed % 10 == 0:
                    print(f"  [LocalLLMReward {completed}/{len(to_request)}] completed")

        return results

    async def judge_async(self, sample: TurnSample) -> RewardResult:
        """Judge a single turn (async)."""
        results = await self.judge_batch_async([sample])
        return results[0]

    def judge_batch(self, samples: List[TurnSample]) -> List[RewardResult]:
        """
        Judge a batch of turns (sync wrapper).

        This is the main entry point for batch reward estimation.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # Already in an event loop - create a new thread to run the coroutine
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.judge_batch_async(samples))
                return future.result()
        else:
            return asyncio.run(self.judge_batch_async(samples))

    async def judge(self, sample: TurnSample) -> RewardResult:
        """Judge a single turn (async interface for compatibility with LLMRewardClient)."""
        return await self.judge_async(sample)

    def judge_sync(self, sample: TurnSample) -> RewardResult:
        """Judge a single turn (sync wrapper)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # Already in an event loop - create a new thread to run the coroutine
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.judge_async(sample))
                return future.result()
        else:
            return asyncio.run(self.judge_async(sample))


# --- Convenience Functions ---

def estimate_reward_local(
    sample: TurnSample,
    config: Optional[LocalLLMRewardConfig] = None,
) -> tuple:
    """
    Synchronous single-sample reward estimation using local LLM.
    Returns (reward, should_update).
    """
    client = LocalLLMRewardClient(config)
    result = client.judge(sample)
    return result.reward, result.should_update


def estimate_rewards_batch_local(
    samples: List[TurnSample],
    config: Optional[LocalLLMRewardConfig] = None,
) -> List[tuple]:
    """
    Synchronous batch reward estimation using local LLM.
    Returns list of (reward, should_update) tuples.
    """
    client = LocalLLMRewardClient(config)
    results = client.judge_batch(samples)
    return [(r.reward, r.should_update) for r in results]
