# Retrieval (transparency)

INDICON 2026 (Paper ID 5232). Development-only retrieval used by the RAG condition and the full
pipeline. Included for transparency; running it live needs a Pinecone index and embedding stack
and is **not** required to reproduce the reported numbers.

## Files
| File | Source | Notes |
|---|---|---|
| `retrieval.py` | `src/retrieval.py` | Two-stream embeddings + MMR reranking. |
| `config.py` | `src/config.py` | Non-secret settings only; all keys read from environment variables. |

## How it works
- **Development-only index.** Only development cases are indexed; **test patients are never added**.
- **Labels withheld.** Eligibility labels are never embedded and are withheld from the retrieved
  text passed to the LLM.
- **Embeddings:** `all-MiniLM-L6-v2` (`config.py: EMBEDDING_MODEL`), dimension 384.
- **Two-stream retrieval:** the patient profile and the concatenated trial criteria are embedded
  separately, each queried against Pinecone, merged by max cosine score per patient, then reranked.
- **MMR reranking:** `lambda = 0.6`, `top_k = 5` (fetch `max(top_k*3, 10)` candidates first);
  `mmr_score = λ·relevance − (1−λ)·max_similarity`.

## Secrets
`config.py` contains **no hardcoded credentials** — `GROQ_API_KEY` and `PINECONE_API_KEY` are read
from environment variables and default to empty strings. `INDEX_NAME` and `EMBEDDING_MODEL` are
non-secret configuration.
