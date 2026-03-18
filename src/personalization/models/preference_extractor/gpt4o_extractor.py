from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from openai import OpenAI
from personalization.config.settings import LocalModelsConfig
from personalization.models.preference_extractor.base import PreferenceExtractorBase as PreferenceExtractor
from personalization.retrieval.preference_store.schemas import (
    ChatTurn,
    PreferenceList,
    preference_list_json_schema,
)


class GPT4OExtractor(PreferenceExtractor):
    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
        # Load system prompt template
        template_path = "fine_tuning_prompt_template.txt"
        if os.path.exists(template_path):
            with open(template_path, "r", encoding="utf-8") as f:
                self.system_prompt = f.read()
        else:
            # Structured prompt that enforces the PreferenceList schema
            self.system_prompt = (
                "You are a preference extraction assistant. "
                "Given a user message, extract any user preferences as condition-action rules.\n\n"
                "Return a JSON object with exactly this structure:\n"
                '{"preferences": [{"condition": "<when this applies>", "action": "<what to do>", "confidence": <0.0-1.0>}]}\n\n'
                "Examples of preferences:\n"
                '- {"condition": "general", "action": "respond in Chinese", "confidence": 0.9}\n'
                '- {"condition": "when writing code", "action": "use Python with type hints", "confidence": 0.8}\n'
                '- {"condition": "when explaining math", "action": "show step-by-step derivation", "confidence": 0.7}\n\n'
                "If no preferences are found, return {\"preferences\": []}.\n"
                "IMPORTANT: The output MUST be a JSON object with a \"preferences\" key containing a list."
            )

    @classmethod
    def from_config(cls, cfg: LocalModelsConfig) -> "GPT4OExtractor":
        # We rely on env var for API key, config for other potential settings if needed
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return cls(api_key=api_key)

    def build_preference_prompt(self, query: str) -> str:
        # GPT4OExtractor uses the system prompt loaded in __init__
        return self.system_prompt

    def _call_kwargs(self, messages):
        """Build kwargs for chat completion, skipping temperature for models that don't support it."""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        # GPT-5 series doesn't support temperature=0
        if not self.model.startswith("gpt-5"):
            kwargs["temperature"] = 0.0
        return kwargs

    def extract_preferences(self, query: str) -> Dict[str, Any]:
        # Reuse logic but return raw dict
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": query},
            ]
            response = self.client.chat.completions.create(**self._call_kwargs(messages))
            content = response.choices[0].message.content
            if content:
                return json.loads(content)
        except Exception as e:
            print(f"Error calling GPT-4o: {e}")
        return {"preferences": []}

    def extract_turn(self, turns) -> PreferenceList:
        # Accept both a single ChatTurn and a list of ChatTurns (history)
        if isinstance(turns, list):
            # Find the last user message in history
            last_user_msg = None
            for t in reversed(turns):
                if hasattr(t, 'role') and t.role == "user":
                    last_user_msg = t.text
                    break
            if not last_user_msg:
                return PreferenceList(preferences=[])
        else:
            # Single ChatTurn
            if turns.role != "user":
                return PreferenceList(preferences=[])
            last_user_msg = turns.text

        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": last_user_msg},
            ]
            response = self.client.chat.completions.create(**self._call_kwargs(messages))
            
            content = response.choices[0].message.content
            if not content:
                return PreferenceList(preferences=[])

            data = json.loads(content)
            return self._parse_to_preference_list(data)

        except Exception as e:
            print(f"Error calling GPT-4o: {e}")
            return PreferenceList(preferences=[])

    @staticmethod
    def _parse_to_preference_list(data: dict) -> PreferenceList:
        """Robustly convert GPT output to PreferenceList, handling non-standard formats."""
        # Best case: already matches schema
        if "preferences" in data and isinstance(data["preferences"], list):
            prefs = []
            for item in data["preferences"]:
                if isinstance(item, dict) and "condition" in item and "action" in item:
                    prefs.append({
                        "condition": str(item["condition"])[:128],
                        "action": str(item["action"])[:256],
                        "confidence": float(item.get("confidence", 0.7)),
                    })
            return PreferenceList.model_validate({"preferences": prefs})

        # GPT returned a flat dict of preferences - convert to condition/action pairs
        prefs = []
        for key, value in data.items():
            if isinstance(value, str) and len(value) > 2:
                prefs.append({
                    "condition": str(key)[:128] if len(str(key)) > 1 else "general",
                    "action": str(value)[:256],
                    "confidence": 0.7,
                })
            elif isinstance(value, dict):
                # Nested dict: try to extract meaningful pairs
                for sub_key, sub_val in value.items():
                    if isinstance(sub_val, str) and len(sub_val) > 2:
                        prefs.append({
                            "condition": str(sub_key)[:128],
                            "action": str(sub_val)[:256],
                            "confidence": 0.7,
                        })
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and len(item) > 2:
                        prefs.append({
                            "condition": str(key)[:128],
                            "action": str(item)[:256],
                            "confidence": 0.7,
                        })

        return PreferenceList.model_validate({"preferences": prefs[:20]})

    def extract_session(self, turns: List[ChatTurn]) -> List[PreferenceList]:
        results = []
        for turn in turns:
            results.append(self.extract_turn(turn))
        return results

