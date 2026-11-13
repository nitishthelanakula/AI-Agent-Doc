"""
retrieval.py

Disha — Three Retrieval Patterns
===================================
All three functions take a LangChain Chroma vectorstore and a question,
and return a list of (Document, score) pairs — though what "score" means
differs slightly per pattern, since each answers a different question:

  top_k      -> "give me the K most similar chunks, whatever the weakest
                 match's score is"
  mmr        -> "give me K chunks similar to the question AND different
                 from each other" (Maximal Marginal Relevance — reduces
                 near-duplicate chunks crowding out variety)
  threshold  -> "give me chunks ONLY if they're similar enough to trust —
                 otherwise give me nothing at all"
"""


def retrieve_top_k(vectorstore, question: str, k: int = 3):
    """Simplest pattern: just the K closest chunks, scores included."""
    return vectorstore.similarity_search_with_relevance_scores(question, k=k)


def retrieve_mmr(vectorstore, question: str, k: int = 3, fetch_k: int = 10, lambda_mult: float = 0.5):
    """Maximal Marginal Relevance: balances relevance to the question
    against diversity among the results themselves. lambda_mult=1.0 is
    pure relevance (behaves like top_k); lambda_mult=0.0 is pure
    diversity. MMR doesn't produce a directly comparable relevance score,
    so we return None in that slot."""
    docs = vectorstore.max_marginal_relevance_search(
        question, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
    )
    return [(doc, None) for doc in docs]


def retrieve_threshold(vectorstore, question: str, k: int = 5, min_score: float = 0.3):
    """Retrieve up to K chunks, but DROP any below min_score. This is the
    pattern that most directly enables an 'I don't know' response — if
    nothing clears the bar, nothing comes back."""
    results = vectorstore.similarity_search_with_relevance_scores(question, k=k)
    return [(doc, score) for doc, score in results if score >= min_score]


PATTERNS = {
    "topk": retrieve_top_k,
    "mmr": retrieve_mmr,
    "threshold": retrieve_threshold,
}


def retrieve(vectorstore, question: str, pattern: str = "topk", **kwargs):
    """Single entry point — dispatches to whichever pattern you name."""
    if pattern not in PATTERNS:
        raise ValueError(f"Unknown retrieval pattern: {pattern}. "
                          f"Choose from: {list(PATTERNS.keys())}")
    return PATTERNS[pattern](vectorstore, question, **kwargs)
