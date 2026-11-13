"""
chunking.py

Disha — Five Chunking Strategies
===================================
Every strategy below takes one document's text and returns a list of
smaller text chunks. All five are called the same way from
build_vectorstore.py:

    chunks = chunk_text(text, strategy="fixed")

...so switching strategies for a live comparison is a one-word change,
not a rewrite.

A note on "semantic" chunking: true semantic chunking uses embeddings to
detect where the TOPIC shifts inside a document, splitting there instead
of at a fixed size. That's more machinery than fits in one teaching
script. Here, semantic_chunk() approximates it with a simple heuristic —
splitting at paragraph (blank-line) boundaries, which in a well-written
policy document usually DOES line up with a topic change. Good enough to
feel the difference from fixed-size chunking; not a substitute for the
real thing in production.
"""
import re


def fixed_chunk(text: str, chunk_size: int = 500) -> list[str]:
    """Split into equal-size pieces, no overlap. Simplest possible strategy —
    and the one most likely to cut a sentence or idea in half."""
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return [c.strip() for c in chunks if c.strip()]


def overlap_chunk(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Same as fixed, but each chunk shares `overlap` characters with the
    next one, so an idea that falls on a boundary isn't lost entirely."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def sentence_chunk(text: str, max_sentences: int = 4) -> list[str]:
    """Split into sentences first, then group every `max_sentences`
    together — never cuts a sentence in half, unlike fixed-size chunking."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if s]
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i:i + max_sentences])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks


def heading_aware_chunk(text: str) -> list[str]:
    """Split at lines that look like headings — numbered sections
    ('1. Scope', '2. Maximum Transmit Power') OR labelled clauses
    ('Clause 4: Service Availability') — so each chunk stays a complete,
    self-contained section rather than an arbitrary slice."""
    heading_pattern = re.compile(
        r"^\s*(\d+\.\s+[A-Z].{0,80}|(?:Clause|Section)\s+\d+:.{0,80})\s*$",
        re.MULTILINE,
    )
    matches = list(heading_pattern.finditer(text))

    if not matches:
        # No numbered headings found — fall back to paragraph splitting.
        return semantic_chunk(text)

    chunks = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[start:end].strip()
        if section:
            chunks.append(section)
    return chunks


def semantic_chunk(text: str) -> list[str]:
    """Approximate semantic chunking by splitting at paragraph (blank-line)
    boundaries. See the module docstring for the honest caveat on this."""
    paragraphs = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in paragraphs if p.strip()]


STRATEGIES = {
    "fixed": fixed_chunk,
    "overlap": overlap_chunk,
    "sentence": sentence_chunk,
    "heading_aware": heading_aware_chunk,
    "semantic": semantic_chunk,
}


def chunk_text(text: str, strategy: str = "overlap", **kwargs) -> list[str]:
    """Single entry point — dispatches to whichever strategy you name."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown chunking strategy: {strategy}. "
                          f"Choose from: {list(STRATEGIES.keys())}")
    return STRATEGIES[strategy](text, **kwargs)
