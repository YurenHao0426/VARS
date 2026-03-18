from __future__ import annotations

import json
import re
import os
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from personalization.config.registry import choose_dtype, choose_device_map
from personalization.config.settings import LocalModelsConfig
from .base import PreferenceExtractor
from personalization.retrieval.preference_store.schemas import (
    PreferenceList,
    preference_list_json_schema,
    ChatTurn,
)

# Hardcoded System Prompt to match SFT training
# This MUST match what was used in training (scripts/split_train_test.py)
SFT_SYSTEM_PROMPT = (
    "Extract user preferences from the query into JSON format based on the PreferenceList schema. "
    "If no preferences are found, return {\"preferences\": []}."
)

class QwenRuleExtractor(PreferenceExtractor):
    """
    Extractor using a Fine-Tuned (SFT) Qwen model.
    Despite the name 'RuleExtractor' (legacy), this now performs direct End-to-End extraction.
    """
    def __init__(self, model_path: str, dtype: torch.dtype, device_map: str = "auto") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=True, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @classmethod
    def from_config(cls, cfg: LocalModelsConfig) -> "QwenRuleExtractor":
        spec = cfg.preference_extractor
        dtype = choose_dtype(spec.dtype)
        device_map = choose_device_map(spec.device_map)
        return cls(spec.local_path, dtype=dtype, device_map=device_map)

    def build_preference_prompt(self, query: str) -> str:
        """
        Construct the prompt string using the tokenizer's chat template.
        Matches the format seen during SFT training.
        """
        messages = [
            {"role": "system", "content": SFT_SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt

    @torch.inference_mode()
    def extract_preferences(self, query: str) -> Dict[str, Any]:
        """
        Directly extract preferences from query using the SFT model.
        Returns a dict compatible with PreferenceList model (key: 'preferences').
        """
        prompt = self.build_preference_prompt(query)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            do_sample=False,        # Deterministic greedy decoding
            max_new_tokens=512,     # Allow enough space for JSON
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        input_len = inputs["input_ids"].shape[1]
        gen_ids = outputs[0][input_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        
        if os.getenv("PREF_DEBUG") == "1":
            print(f"[debug][extractor] Raw output: {text}")

        # Try parsing JSON
        try:
            # 1. Direct parse
            data = json.loads(text)
            
            # 2. Validate against schema structure
            validated = PreferenceList.model_validate(data)
            return validated.model_dump()
            
        except Exception:
            # Fallback: Try to find JSON blob if model outputted extra text (rare for SFT but possible)
            extracted_json = self._extract_json_substring(text)
            if extracted_json:
                try:
                    data = json.loads(extracted_json)
                    validated = PreferenceList.model_validate(data)
                    return validated.model_dump()
                except:
                    pass
            
            # If all fails, return empty
            return {"preferences": []}

    def _extract_json_substring(self, text: str) -> str | None:
        """Helper to find { ... } block in text."""
        # Find first '{' and last '}'
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        return None

    @torch.inference_mode()
    def batch_extract_preferences(self, queries: List[str], batch_size: int = 64) -> List[Dict[str, Any]]:
        """
        Batch extract preferences from multiple queries using left-padded batching.
        """
        if not queries:
            return []

        # Save and set padding side for decoder-only batched generation
        orig_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        all_results = []
        prompts = [self.build_preference_prompt(q) for q in queries]

        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start:start + batch_size]
            inputs = self.tokenizer(
                batch_prompts, return_tensors="pt", padding=True, truncation=True
            ).to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=512,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            for i in range(len(batch_prompts)):
                input_len = (inputs["attention_mask"][i] == 1).sum().item()
                gen_ids = outputs[i][input_len:]
                text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

                try:
                    data = json.loads(text)
                    validated = PreferenceList.model_validate(data)
                    all_results.append(validated.model_dump())
                except Exception:
                    extracted_json = self._extract_json_substring(text)
                    if extracted_json:
                        try:
                            data = json.loads(extracted_json)
                            validated = PreferenceList.model_validate(data)
                            all_results.append(validated.model_dump())
                            continue
                        except Exception:
                            pass
                    all_results.append({"preferences": []})

        self.tokenizer.padding_side = orig_padding_side
        return all_results

    def extract_turn(self, turns: List[ChatTurn]) -> PreferenceList:
        """
        Extract preferences from the LAST user turn in the history.
        We don't concat history because our SFT model was trained on single-turn extraction.
        Using context might confuse it unless we trained it that way.
        """
        # Find the last user message
        last_user_msg = None
        for t in reversed(turns):
            if t.role == "user":
                last_user_msg = t.text
                break
        
        if not last_user_msg:
            return PreferenceList(preferences=[])
        
        result_dict = self.extract_preferences(last_user_msg)
        return PreferenceList.model_validate(result_dict)

    def extract_session(self, turns: List[ChatTurn]) -> List[PreferenceList]:
        """
        Extract preferences from ALL user turns individually.
        """
        results = []
        for turn in turns:
            if turn.role == "user":
                res = self.extract_preferences(turn.text)
                results.append(PreferenceList.model_validate(res))
            else:
                results.append(PreferenceList(preferences=[]))
        return results
