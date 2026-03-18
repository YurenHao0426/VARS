"""
vLLM-based ChatModel implementation for high-throughput inference.

This provides the same interface as LlamaChatModel but uses vLLM HTTP API
for much faster inference (3000+ sessions/hr vs 20 sessions/hr).
"""

from typing import List, Optional
import time
import requests

from personalization.models.llm.base import ChatModel
from personalization.types import ChatTurn


class VLLMChatModel(ChatModel):
    """
    ChatModel implementation using vLLM HTTP API.

    This is a drop-in replacement for LlamaChatModel that uses vLLM
    for much faster inference.
    """

    def __init__(
        self,
        vllm_url: str = "http://localhost:8003/v1",
        model_name: str = None,
        max_context_length: int = 8192,
        timeout: int = 120,
    ):
        self.vllm_url = vllm_url.rstrip('/')
        self.model_name = model_name
        self.max_context_length = max_context_length
        self.timeout = timeout

        # Discover model name if not provided
        if self.model_name is None:
            self._discover_model()

    def _discover_model(self):
        """Discover the model name from the vLLM server."""
        max_retries = 30
        for attempt in range(max_retries):
            try:
                response = requests.get(f"{self.vllm_url}/models", timeout=10)
                response.raise_for_status()
                models = response.json()
                if models.get("data") and len(models["data"]) > 0:
                    self.model_name = models["data"][0]["id"]
                    return
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt * 0.5, 10)
                    time.sleep(wait_time)

        # Fallback
        self.model_name = "default"
        print(f"[VLLMChatModel] Warning: Could not discover model, using '{self.model_name}'")

    def health_check(self) -> bool:
        """Check if the vLLM server is healthy."""
        try:
            response = requests.get(f"{self.vllm_url.replace('/v1', '')}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using character-based heuristic.

        For Llama models, ~4 characters per token is a reasonable estimate.
        We use 3.5 to be conservative (slightly overestimate tokens).
        """
        return int(len(text) / 3.5)

    def _build_messages(
        self,
        history: List[ChatTurn],
        memory_notes: List[str],
        max_new_tokens: int = 512,
        global_notes: List[str] = None,
    ) -> List[dict]:
        """Build messages list for chat completion API with auto-truncation.

        If the context exceeds max_context_length, older conversation turns
        are removed to keep only the most recent context that fits.

        Args:
            global_notes: If provided, these are always-applicable preferences
                displayed in a separate section from task-specific retrieved notes.
        """
        # Use CollaborativeAgents-style system prompt
        has_any_notes = memory_notes or global_notes
        if has_any_notes:
            # Build preference sections
            pref_sections = ""
            if global_notes:
                global_bullet = "\n".join(f"- {n}" for n in global_notes)
                pref_sections += f"## General Preferences (always apply)\n{global_bullet}\n\n"
            if memory_notes:
                task_bullet = "\n".join(f"- {n}" for n in memory_notes)
                if global_notes:
                    pref_sections += f"## Task-Specific Preferences\n{task_bullet}\n"
                else:
                    pref_sections += f"{task_bullet}\n"

            system_content = (
                "You are a collaborative AI agent helping users solve writing, question answering, math, and coding problems.\n\n"
                "# User Preferences\n"
                "The user has a set of preferences for how you should behave. If you do not follow these preferences, "
                "the user will be unable to learn from your response and you will need to adjust your response to adhere "
                "to these preferences (so it is best to follow them initially).\n\n"
                "**IMPORTANT**: If the user explicitly requests something in THIS conversation (e.g., asks you to change "
                "your format, style, or approach), that request takes PRIORITY over the remembered preferences below. "
                "Always adapt to the user's direct feedback first.\n\n"
                "Based on your past interactions with the user, you have maintained a set of notes about the user's preferences:\n"
                f"{pref_sections}\n"
                "# Before Responding\n"
                "Before writing your response, briefly consider:\n"
                "1. Which preferences above are relevant to this specific request?\n"
                "2. How will you satisfy each relevant preference in your response?\n\n"
                "# Conversation Guidelines:\n"
                "- If the user asks you to adjust your response (e.g., 'be more concise', 'focus on intuition'), you MUST change your approach accordingly. Do NOT repeat the same response.\n"
                "- If the user's message is unclear, lacks details, or is ambiguous (e.g. length of an essay, format requirements, "
                "specific constraints), do not make assumptions. Ask for clarification and ensure you have enough information before providing an answer.\n"
                "- Your goal is to help the user solve their problem. Adhere to their preferences and do your best to help them solve their problem.\n"
                "- **Verify**: Before finalizing, check that your response satisfies the relevant preferences listed above.\n"
            )
        else:
            # Vanilla mode - no preferences
            system_content = (
                "You are a collaborative AI agent helping users solve writing, question answering, math, and coding problems.\n\n"
                "# Conversation Guidelines:\n"
                "- If the user's message is unclear, lacks details, or is ambiguous (e.g. length of an essay, format requirements, "
                "specific constraints), do not make assumptions. Ask for clarification and ensure you have enough information before providing an answer.\n"
                "- Your goal is to help the user solve their problem. Do your best to help them.\n"
            )
        system_message = {"role": "system", "content": system_content}

        # Calculate available tokens for conversation history
        # Reserve space for: system prompt + max_new_tokens + safety margin
        system_tokens = self._estimate_tokens(system_content)
        available_tokens = self.max_context_length - system_tokens - max_new_tokens - 100  # 100 token safety margin

        # Build conversation messages from history
        conversation_messages = []
        for turn in history:
            conversation_messages.append({"role": turn.role, "content": turn.text})

        # Check if truncation is needed
        total_conv_tokens = sum(self._estimate_tokens(m["content"]) for m in conversation_messages)

        if total_conv_tokens > available_tokens:
            # Truncate from the beginning (keep recent messages)
            truncated_messages = []
            current_tokens = 0

            # Iterate from most recent to oldest
            for msg in reversed(conversation_messages):
                msg_tokens = self._estimate_tokens(msg["content"])
                if current_tokens + msg_tokens <= available_tokens:
                    truncated_messages.insert(0, msg)
                    current_tokens += msg_tokens
                else:
                    # Stop adding older messages
                    break

            conversation_messages = truncated_messages
            if len(truncated_messages) < len(history):
                print(f"[VLLMChatModel] Truncated context: kept {len(truncated_messages)}/{len(history)} turns "
                      f"({current_tokens}/{total_conv_tokens} estimated tokens)")

        messages = [system_message] + conversation_messages
        return messages

    def build_messages(
        self,
        history: List[ChatTurn],
        memory_notes: List[str],
        max_new_tokens: int = 512,
        global_notes: List[str] = None,
    ) -> List[dict]:
        """Public method to build messages without calling the API.

        Used for batch processing where messages are collected first,
        then sent in batch to vLLM for concurrent processing.
        """
        return self._build_messages(history, memory_notes, max_new_tokens, global_notes=global_notes)

    def answer(
        self,
        history: List[ChatTurn],
        memory_notes: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
    ) -> str:
        """Generate a response using vLLM HTTP API."""
        messages = self._build_messages(history, memory_notes, max_new_tokens)

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        # Retry with exponential backoff
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.vllm_url}/chat/completions",
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                elif response.status_code == 400:
                    error_text = response.text
                    # Handle context length error
                    if "max_tokens" in error_text and max_new_tokens > 64:
                        payload["max_tokens"] = max(64, max_new_tokens // 2)
                        continue
                    raise RuntimeError(f"vLLM error: {error_text[:200]}")
                else:
                    raise RuntimeError(f"vLLM HTTP {response.status_code}: {response.text[:200]}")

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError("vLLM request timeout")
            except requests.exceptions.ConnectionError as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"vLLM connection error: {e}")

        raise RuntimeError("Max retries exceeded for vLLM request")
