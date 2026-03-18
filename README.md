# VARS: Vector-Adapted Retrieval Scoring for Personalized LLM Assistants

> **User Preference Modeling for Conversational LLM Agents: Weak Rewards from Retrieval-Augmented Interaction**
>
> Yuren Hao, Shuhaib Mehri, ChengXiang Zhai, Dilek Hakkani-Tür
>
> University of Illinois at Urbana-Champaign
>
> [[Paper]](paper/paper.pdf)

## Overview

Large language models are increasingly used as conversational collaborators, yet most lack a persistent user model, forcing users to repeatedly restate preferences across sessions. **VARS** (Vector-Adapted Retrieval Scoring) is a pipeline-agnostic, frozen-backbone framework that represents each user with long-term and short-term vectors in a shared preference space and uses these vectors to bias retrieval scoring over structured preference memory. The vectors are updated online from weak scalar feedback, enabling personalization without per-user fine-tuning.

### Key Idea

At each turn, the system:
1. **Extracts** structured (condition, action) preferences from dialogue via a lightweight finetuned model
2. **Stores** preferences as memory cards with dense embeddings in a FAISS index
3. **Retrieves** relevant preferences via dense search + cross-encoder reranking, biased by a user-specific vector bonus
4. **Updates** dual user vectors (long-term + short-term) online via REINFORCE from keyword-based reward signals

The effective user vector at turn *t* combines stable cross-session identity with transient within-session context:

```
z_eff = β_L · z_long + β_S · z_short
```

## Results

Evaluated on [MultiSessionCollab](https://github.com/shmehri/MultiSessionCollab) (60 profiles × 60 sessions, 3,600 sessions per method) across math and code tasks:

| Method | Success (%) ↑ | Timeout (%) ↓ | User tokens ↓ |
|--------|:---:|:---:|:---:|
| Vanilla | 54.3 | 29.2 | 232.9 |
| Contextual | 52.4 | 31.4 | 213.7 |
| All-memory | 50.9 | 33.4 | 226.8 |
| Reflection | 54.4 | 28.8 | 207.5 |
| RAG | 52.0 | 44.3 | **188.4** |
| **VARS** | **55.2** | **26.4** | 193.6 |

VARS achieves the strongest overall performance, matches Reflection in task success while significantly reducing timeout rate (-2.4 pp, *p*=0.046) and user effort (-13.9 tokens, *p*=0.021). The learned long-term vectors align with cross-user preference overlap (*p*=0.006), while short-term vectors capture session-specific adaptation.

## Architecture

```
User Query u_t ──► Dense Retrieval ──► Reranker ──► User-Aware Scoring ──► Top-J notes ──► LLM Response
                        │                               s(u,m;U) = s_0 + ⟨z_eff, v_m⟩
                        │                                      ▲
                  Preference                              User Vector
                  Memory (FAISS)                     z_eff = β_L·z_L + β_S·z_S
                        ▲                                      ▲
                        │                              REINFORCE Update
                  Preference Extractor                 from keyword reward r̂_t
                  M_ext (Qwen3-0.6B)
```

### Core Modules

| Module | Description |
|--------|-------------|
| `serving/personalized_llm.py` | Main inference interface (`chat()`, `chat_prepare()`, `chat_complete()`) |
| `models/llm/vllm_chat.py` | vLLM HTTP client for high-throughput batched inference |
| `models/embedding/qwen3_8b.py` | Dense embedding (Qwen3-Embedding-8B) |
| `models/reranker/qwen3_reranker.py` | Cross-encoder reranking (Qwen3-Reranker-8B) |
| `models/preference_extractor/` | Lightweight preference extraction from conversation |
| `retrieval/pipeline.py` | RAG retrieval pipeline with FAISS vector store and PCA item space |
| `user_model/policy/reinforce.py` | REINFORCE policy for dual user-vector optimization |
| `feedback/reward_model.py` | Keyword-based reward heuristic |
| `feedback/handlers.py` | Retrieval-attribution gating and online RL updates |

## Models and Data

### Preference Extractor

A 0.6B-parameter Qwen3 model finetuned for structured preference extraction. Given a dialogue window, it outputs JSON preference tuples `{condition, action, confidence}`.

- **Model**: [blackhao0426/pref-extractor-qwen3-0.6b-full-sft](https://huggingface.co/blackhao0426/pref-extractor-qwen3-0.6b-full-sft)
- **Training Data**: [blackhao0426/user-preference-564k](https://huggingface.co/datasets/blackhao0426/user-preference-564k) — 564K examples constructed from public chat logs (LMSYS-Chat, WildChat), instruction-tuning corpora (Alpaca, SlimOrca), and GPT-5.1-labeled preference JSON

On a held-out set, the extractor achieves 99.7% JSON validity and 97.5% recall at 37.7% precision (intentionally high-recall; downstream reranker and user vector filter irrelevant cards).

### Backbone Models

| Role | Model | Parameters |
|------|-------|------------|
| Agent LLM | Llama-3.1-8B-Instruct (via vLLM) | 8B |
| User Simulator | Llama-3.3-70B-Instruct (via vLLM) | 70B |
| Dense Embedding | Qwen3-Embedding-8B | 8B |
| Reranker | Qwen3-Reranker-8B | 8B |
| Preference Extractor | Qwen3-0.6B (finetuned) | 0.6B |

All backbone components are kept frozen; online adaptation occurs only through the per-user vectors.

## Installation

```bash
pip install -e .
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.3.0
- Transformers >= 4.44.0

## Quick Start

```python
from personalization.serving import PersonalizedLLM

llm = PersonalizedLLM.from_config("configs/local_models.yaml")

# Multi-turn personalized chat
response = llm.chat(user_id="user_001", query="Explain quicksort")

# The system automatically:
# 1. Extracts preferences from conversation history
# 2. Retrieves relevant preferences via dense retrieval + reranking
# 3. Adds user-vector bonus to retrieval scores
# 4. Augments the LLM prompt with top-ranked preference notes
# 5. Updates user vectors from implicit feedback (REINFORCE)
```

## Citation

```bibtex
@article{hao2025vars,
  title={User Preference Modeling for Conversational LLM Agents: Weak Rewards from Retrieval-Augmented Interaction},
  author={Hao, Yuren and Mehri, Shuhaib and Zhai, ChengXiang and Hakkani-T{\"u}r, Dilek},
  year={2025}
}
```

## License

Apache-2.0
