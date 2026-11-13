"""
context.py

Disha — Context Assembly: Evidence Packing & Deduplication
==============================================================
Retrieval can return near-duplicate chunks (e.g., the same fact split two
overlapping ways) and can return more text than is worth spending tokens
on. This module runs on every retrieval result BEFORE it reaches the LLM:

  1. Deduplicate near-identical chunks (simple normalised-text comparison)
  2. Pack whatever remains into a fixed character budget, so cost and
     latency stay predictable no matter how much was retrieved
"""


def normalise(text: str) -> str:
    return " ".join(text.lower().split())


def deduplicate(results: list) -> list:
    """Drop chunks that are near-identical to one already kept. Good
    enough for overlapping chunks of the same source text, without
    needing a similarity model of its own."""
    seen_normalised = []
    deduped = []
    for doc, score in results:
        norm = normalise(doc.page_content)
        is_duplicate = any(norm in seen or seen in norm for seen in seen_normalised)
        if not is_duplicate:
            seen_normalised.append(norm)
            deduped.append((doc, score))
    return deduped


def pack_context(results: list, max_chars: int = 3000) -> str:
    """Build the final context string handed to the LLM, citing each
    chunk's source, and stopping once the character budget runs out."""
    pieces = []
    total_chars = 0
    for doc, score in results:
        source = doc.metadata.get("source", "unknown")
        piece = f"(From {source})\n{doc.page_content}"
        if total_chars + len(piece) > max_chars:
            break
        pieces.append(piece)
        total_chars += len(piece)
    return "\n\n---\n\n".join(pieces)


def assemble_context(results: list, max_chars: int = 3000):
    """Full pipeline: dedupe, then pack. Returns the context string AND
    the final (doc, score) list actually used, so the caller can still
    report accurate sources for whatever made it into the prompt."""
    deduped = deduplicate(results)
    context_text = pack_context(deduped, max_chars=max_chars)
    return context_text, deduped
