"""
rag_engine/policy_rag_tool.py
-----------------------------
Retrieval-Augmented Generation tool for corporate policy lookup.

Chunks the company_policy.md document into logical sections and performs
semantic similarity search to find relevant policy clauses for a given query.

Uses TF-IDF vectorization for lightweight semantic matching (no external
vector DB or embedding API required for the prototype).

Main function:
    check_company_policy(query: str) -> str

Returns the most relevant policy sections as plain text that the LLM agent
can read and reason over.
"""

import os
import re
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Path to policy document
POLICY_PATH = os.path.join(os.path.dirname(__file__), "company_policy.md")

# Cache for loaded and chunked policy
_policy_chunks: List[dict] = []
_vectorizer = None
_tfidf_matrix = None


def _load_and_chunk_policy() -> List[dict]:
    """
    Load company_policy.md and split it into logical sections.

    Each chunk contains:
      - section_title: The heading of the section
      - content: The full text of the section
    """
    with open(POLICY_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    # Split on markdown headings (## or ###)
    sections = re.split(r'\n(?=## )', text)

    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Extract title from first line
        lines = section.split("\n")
        title = lines[0].lstrip("#").strip()
        content = section

        # Also split subsections (###) within each section
        subsections = re.split(r'\n(?=### )', section)
        if len(subsections) > 1:
            for sub in subsections:
                sub = sub.strip()
                if sub:
                    sub_lines = sub.split("\n")
                    sub_title = sub_lines[0].lstrip("#").strip()
                    chunks.append({
                        "section_title": f"{title} > {sub_title}" if sub_title != title else title,
                        "content": sub,
                    })
        else:
            chunks.append({
                "section_title": title,
                "content": content,
            })

    return chunks


def _initialize():
    """Initialize the TF-IDF vectorizer and index the policy chunks."""
    global _policy_chunks, _vectorizer, _tfidf_matrix

    if _policy_chunks:
        return  # Already initialized

    _policy_chunks = _load_and_chunk_policy()

    # Build TF-IDF index over chunk content
    texts = [chunk["content"] for chunk in _policy_chunks]
    _vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
    )
    _tfidf_matrix = _vectorizer.fit_transform(texts)

    print(f"[policy_rag] Indexed {len(_policy_chunks)} policy sections.")


def check_company_policy(query: str, top_k: int = 3) -> str:
    """
    Search the company policy document for sections relevant to a query.

    Uses TF-IDF cosine similarity for lightweight semantic matching.

    Args:
        query: Natural language question, e.g.
               "What is the laptop budget limit for Engineering?"
               "Do we need approval for Apple products?"
               "What are the vendor payment terms?"
        top_k: Number of top matching sections to return

    Returns:
        A string containing the most relevant policy sections,
        formatted for the LLM agent to read and reason over.
    """
    _initialize()

    # Vectorize the query
    query_vec = _vectorizer.transform([query])

    # Compute similarity scores
    similarities = cosine_similarity(query_vec, _tfidf_matrix)[0]

    # Get top-k indices sorted by score
    top_indices = similarities.argsort()[-top_k:][::-1]

    # Build response
    results = []
    for idx in top_indices:
        score = similarities[idx]
        if score < 0.01:  # Skip completely irrelevant sections
            continue
        chunk = _policy_chunks[idx]
        results.append(
            f"--- Policy Section: {chunk['section_title']} (relevance: {score:.2f}) ---\n"
            f"{chunk['content']}"
        )

    if not results:
        return "No relevant policy sections found for this query."

    return "\n\n".join(results)


# ---------------------------------------------------------------------------
# Quick test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[policy_rag] Testing policy search...\n")

    queries = [
        "What is the spending limit for Engineering department?",
        "Can we buy Apple MacBooks?",
        "What are the vendor payment terms?",
        "What happens in an emergency purchase?",
        "What is the cash reserve policy?",
    ]

    for q in queries:
        print(f"Q: {q}")
        print(f"A:\n{check_company_policy(q, top_k=2)}")
        print("\n" + "=" * 70 + "\n")
