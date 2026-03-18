from typing import List, Protocol, Optional
from personalization.types import ChatTurn

class ChatModel(Protocol):
    def answer(
        self,
        history: List[ChatTurn],
        memory_notes: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
    ) -> str:
        """
        Generate an assistant response given conversation history and memory notes.

        Args:
            history: The conversation history ending with the current user turn.
            memory_notes: List of retrieved memory content strings.
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling.
            top_k: Top-k sampling.

        Returns:
            The generated assistant response text.
        """
        ...

