# VARS: Vector-Augmented Retrieval System for Personalized LLM Assistants

VARS is a personalization framework that enables LLM assistants to learn and adapt to individual user preferences over multi-session interactions. It combines **dense retrieval**, **reranking**, and **REINFORCE-based user vector learning** to deliver personalized responses without explicit user configuration.

## Architecture

```
User Query ──► Preference Retrieval (Dense + Rerank) ──► Augmented Prompt ──► LLM Response
                       ▲                                                          │
                       │                                                          ▼
                 User Vector ◄──── REINFORCE Update ◄──── Implicit Feedback Signal
                       ▲
                       │
              Preference Extractor ◄──── Conversation History
```

### Core Components

| Module | Description |
|--------|-------------|
| `serving/personalized_llm.py` | Main inference interface (`chat()`, `chat_prepare()`, `chat_complete()`) |
| `models/llm/vllm_chat.py` | vLLM HTTP client for high-throughput batched inference |
| `models/embedding/qwen3_8b.py` | Dense embedding (Qwen3-Embedding-8B) |
| `models/reranker/qwen3_reranker.py` | Cross-encoder reranking (Qwen3-Reranker-8B) |
| `models/preference_extractor/` | Online preference extraction from conversation |
| `retrieval/pipeline.py` | RAG retrieval pipeline with FAISS vector store |
| `user_model/policy/reinforce.py` | REINFORCE policy for user vector optimization |
| `feedback/` | Reward model (keyword / LLM judge) and online RL updates |

## Models

### Preference Extractor

We fine-tuned a Qwen3-0.6B model for structured preference extraction from conversational context.

- **Model**: [blackhao0426/pref-extractor-qwen3-0.6b-full-sft](https://huggingface.co/blackhao0426/pref-extractor-qwen3-0.6b-full-sft)
- **Training Data**: [blackhao0426/user-preference-564k](https://huggingface.co/datasets/blackhao0426/user-preference-564k) (564K examples of user preference extraction)

The extractor takes conversation turns as input and outputs structured `{condition, action, confidence}` preference tuples.

### Other Models Used

| Role | Model |
|------|-------|
| Agent LLM | LLaMA-3.1-8B-Instruct (via vLLM) |
| Dense Embedding | Qwen3-Embedding-8B |
| Reranker | Qwen3-Reranker-8B |
| Reward Judge | LLaMA-3.1-8B-Instruct or GPT-4o-mini |

## Installation

```bash
pip install -e .
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.3.0
- Transformers >= 4.44.0

## Usage

```python
from personalization.serving import PersonalizedLLM

llm = PersonalizedLLM.from_config("configs/local_models.yaml")

# Multi-turn personalized chat
response = llm.chat(user_id="user_001", query="Explain quicksort")

# The system automatically:
# 1. Extracts preferences from conversation history
# 2. Retrieves relevant preferences via dense retrieval + reranking
# 3. Augments the prompt with personalized context
# 4. Updates user vector from implicit feedback (REINFORCE)
```

## License

Apache-2.0
