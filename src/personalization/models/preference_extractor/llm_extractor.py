from typing import List, Dict, Any
import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

from personalization.models.preference_extractor.base import PreferenceExtractorBase
from personalization.retrieval.preference_store.schemas import ChatTurn, PreferenceList
from personalization.config.settings import LocalModelsConfig
from personalization.config.registry import choose_dtype, choose_device_map

class PreferenceExtractorLLM(PreferenceExtractorBase):
    def __init__(
        self,
        model_path: str,
        prompt_template_path: str = "fine_tuning_prompt_template.txt",
        device_map: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 512,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.max_new_tokens = max_new_tokens
        
        if os.path.exists(prompt_template_path):
            with open(prompt_template_path, "r", encoding="utf-8") as f:
                self.prompt_template = f.read()
        else:
            print(f"Warning: Prompt template not found at {prompt_template_path}. Using fallback.")
            self.prompt_template = "Extract user preferences from the following conversation."

    @classmethod
    def from_config(cls, cfg: LocalModelsConfig, name: str = "qwen3_0_6b_sft") -> "PreferenceExtractorLLM":
        # We need to access the specific extractor config by name
        # Assuming cfg has a way to access extra configs or we update LocalModelsConfig to support multiple extractors
        # For now, let's look for it in the 'preference_extractor' dict if it was a Dict, but it is a ModelSpec.
        # We need to update LocalModelsConfig to support a dictionary of extractors or a specific one.
        # Based on user design doc:
        # preference_extractor:
        #   qwen3_0_6b_sft: ...
        
        # We might need to manually parse the raw config or update settings.py
        # Let's assume settings.py will be updated to hold a map or specific fields.
        # For now, if we use the existing ModelSpec for preference_extractor in cfg, we assume it points to this model.
        
        # BUT the design doc says "preference_extractor" in local_models.yaml will have "qwen3_0_6b_sft" key.
        # The current settings.py defines preference_extractor as a single ModelSpec.
        # We will need to update settings.py first to support multiple extractors or a dict.
        # I will proceed implementing this class assuming arguments are passed, and update settings/registry later.
        
        # This from_config might change depending on how settings.py is refactored.
        # For now I will implement it assuming a direct ModelSpec is passed, or we handle it in registry.
        pass
        return None 

    def _build_prompt(self, turns: List[ChatTurn]) -> str:
        # Construct messages list for chat template
        messages = [{"role": "system", "content": self.prompt_template}]
        
        # Window size 6
        window = turns[-6:]
        
        # Add conversation history
        # We need to format the conversation as input context.
        # Since the task is to extract preferences from the *whole* context (or latest turn?),
        # usually we provide the conversation and ask for extraction.
        # But LLaMA-Factory SFT usually expects:
        # System: <template>
        # User: <input>
        # Assistant: <output>
        
        # We should pack the conversation history into the User message?
        # Or if we trained with multi-turn chat format?
        # Assuming "Input" column in dataset was the conversation history.
        
        history_texts = []
        for t in window:
            role = "User" if t.role == "user" else "Assistant"
            history_texts.append(f"{role}: {t.text}")
        
        conversation_text = "\n".join(history_texts)
        
        # Construct the User input
        # We append a trigger instruction if it wasn't part of the training input implicitly.
        # But based on your template, the User Input Example was just the query "I am a Python developer..."
        # So likely we should just feed the conversation text as the user message.
        
        messages.append({"role": "user", "content": conversation_text})
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        return prompt

    def _generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_text[len(prompt):]

    def _parse_preferences(self, raw_output: str) -> PreferenceList:
        start = raw_output.find("{")
        end = raw_output.rfind("}")
        
        if start == -1 or end == -1 or end <= start:
            return PreferenceList(preferences=[])
            
        json_str = raw_output[start:end+1]
        try:
            data = json.loads(json_str)
            return PreferenceList.model_validate(data)
        except Exception:
            return PreferenceList(preferences=[])

    def extract_turn(self, turns: List[ChatTurn]) -> PreferenceList:
        prompt = self._build_prompt(turns)
        raw_output = self._generate(prompt)
        return self._parse_preferences(raw_output)

    # Legacy support
    def build_preference_prompt(self, query: str) -> str:
        # Wrap query in a dummy turn
        turn = ChatTurn(
            user_id="dummy", session_id="dummy", turn_id=0, 
            role="user", text=query
        )
        return self._build_prompt([turn])

    def extract_preferences(self, query: str) -> Dict[str, Any]:
        turn = ChatTurn(
            user_id="dummy", session_id="dummy", turn_id=0, 
            role="user", text=query
        )
        prefs = self.extract_turn([turn])
        return prefs.model_dump()

