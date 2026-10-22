"""
src/retrieval.py
----------------
Phase 2: Hybrid dual-vector retrieval with MMR deduplication.

What changed from Phase 1:
- OLD: one embedding for the full combined query → single Pinecone call
- NEW: two separate embeddings (patient profile + trial criteria) → two
  Pinecone calls → results merged by score → MMR applied to remove
  redundant matches before returning context to the pipeline

Why two embeddings?
  Patient profiles and trial criteria live in different semantic spaces.
  A single combined embedding dilutes both signals. Keeping them separate
  means a patient with a rare FEV1 value will surface trials that mention
  FEV1 thresholds even when the rest of the criteria text is very different.

Why MMR?
  Without it, both Pinecone calls tend to return the same top-1 or top-2
  patients, wasting context window space on duplicates.
"""

from src.embedding import get_embedding
from src.pinecone_db import get_index


# ------------------------------------------------------------------ #
# MMR — Maximal Marginal Relevance                                    #
# ------------------------------------------------------------------ #

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Dot product of two already-normalised unit vectors (sentence-transformers output)."""
    return sum(x * y for x, y in zip(a, b))


def _mmr(
    candidates: list[dict],
    query_embedding: list[float],
    top_k: int,
    lambda_param: float = 0.6
) -> list[dict]:
    """
    Maximal Marginal Relevance selection.

    Balances relevance to the query (high score) against diversity
    from already-selected results (low similarity to selected set).

    lambda_param:
        1.0  → pure relevance (same as no MMR)
        0.0  → pure diversity
        0.6  → default, slightly favours relevance

    Each candidate dict must have keys: id, score, values, metadata.
    """
    if not candidates:
        return []

    selected = []
    remaining = list(candidates)

    while remaining and len(selected) < top_k:
        if not selected:
            # First pick: highest relevance score
            best = max(remaining, key=lambda c: c["score"])
        else:
            best = None
            best_mmr = float("-inf")

            for c in remaining:
                # Relevance term
                relevance = c["score"]

                # Redundancy term: max similarity to already-selected
                max_sim = max(
                    _cosine_similarity(c["values"], s["values"])
                    for s in selected
                )

                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best = c

        selected.append(best)
        remaining.remove(best)

    return selected


# ------------------------------------------------------------------ #
# Dual-vector hybrid retrieval                                        #
# ------------------------------------------------------------------ #

def retrieve_hybrid(
    patient_profile: str,
    inclusion_criteria: str,
    exclusion_criteria: str,
    top_k: int = 5,
    mmr_lambda: float = 0.6
) -> tuple[str, list[str], list[float]]:
    """
    Two-stream retrieval:
      Stream A — embed the patient profile alone
      Stream B — embed the concatenated trial criteria alone

    Results from both streams are merged, de-duplicated by patient_id,
    re-ranked by score, and then MMR is applied to select the final top_k.

    Returns:
        memory_context  — formatted string injected into the LLM prompt
        retrieved_ids   — list of patient_id strings
        retrieved_scores — parallel list[float], the Pinecone cosine score
                           for each id in retrieved_ids (used as one input
                           to the pipeline's combined confidence estimate)
    """
    index = get_index()
    if index is None:
        return "", [], []

    trial_criteria_text = (
        f"Inclusion Criteria:\n{inclusion_criteria or ''}\n\n"
        f"Exclusion Criteria:\n{exclusion_criteria or ''}"
    )

    # Embed each stream independently
    try:
        profile_emb   = get_embedding(str(patient_profile or ""))
        criteria_emb  = get_embedding(trial_criteria_text)
    except Exception as e:
        print(f"[retrieval] Embedding error: {e}")
        return "", [], []

    # Query Pinecone twice — fetch more than top_k so MMR has room to pick
    fetch_k = max(top_k * 3, 10)

    def _query(vector):
        try:
            return index.query(
                vector=vector,
                top_k=fetch_k,
                include_metadata=True,
                include_values=True   # needed for MMR cosine similarity
            )["matches"]
        except Exception as e:
            print(f"[retrieval] Pinecone query error: {e}")
            return []

    stream_a = _query(profile_emb)   # patient-profile stream
    stream_b = _query(criteria_emb)  # trial-criteria stream

    # Merge: keep highest score per patient_id
    merged: dict[str, dict] = {}

    for match in stream_a + stream_b:
        pid   = match.get("id") or match.get("metadata", {}).get("patient_id", "unknown")
        score = match.get("score", 0.0)

        if pid not in merged or score > merged[pid]["score"]:
            merged[pid] = {
                "id":       pid,
                "score":    score,
                "values":   match.get("values", []),
                "metadata": match.get("metadata", {}),
            }

    candidates = sorted(merged.values(), key=lambda x: x["score"], reverse=True)

    # Apply MMR using the profile embedding as the primary query signal
    selected = _mmr(candidates, profile_emb, top_k=top_k, lambda_param=mmr_lambda)

    # Build context string and ID list
    memory_context = ""
    retrieved_ids  = []
    retrieved_scores: list[float] = []

    for match in selected:
        pid      = match["id"]
        metadata = match["metadata"]
        score    = match["score"]

        retrieved_ids.append(pid)
        retrieved_scores.append(float(score))

        memory_context += (
            f"\nPatient Profile:\n{metadata.get('patient_profile', '')}\n"
            # Phase 0 fix: label intentionally excluded from context shown to LLM.
            # It lives in metadata for evaluation purposes only — exposing it here
            # would let the LLM read the answer from similar retrieved cases
            # (indirect label leakage through the context window).
            f"\nRetrieval Score: {score:.4f}\n"
            "-------------------------\n"
        )

    return memory_context, retrieved_ids, retrieved_scores


# ------------------------------------------------------------------ #
# Legacy shim — keeps upload.py / any other caller working           #
# ------------------------------------------------------------------ #

def retrieve_similar(query_text: str, top_k: int = 5) -> list:
    """
    Phase 1 single-vector retrieval — kept for backward compatibility.
    New code should call retrieve_hybrid() directly.
    """
    index = get_index()
    query_emb = get_embedding(query_text)
    results = index.query(
        vector=query_emb,
        top_k=top_k,
        include_metadata=True
    )
    return results["matches"]