"""
pii.py

Disha — PII-Safe Ingestion Basics
====================================
Before any chunk gets embedded and stored, it passes through this
redaction step. This is intentionally simple: a handful of regex
patterns catching the most common PII shapes (emails, phone numbers,
PAN-style IDs). It is NOT a substitute for a proper NER-based PII
detector in a real production pipeline — but it demonstrates the
principle live, on a real example, which is the point of this lab.

Try it yourself: the HTML FAQ document in this lab contains a real
example — a NOC Duty Manager's name, email, and phone number — placed
there specifically so you can watch this function redact it during
ingestion.
"""
import re

PATTERNS = {
    "EMAIL": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "PHONE_INDIA": re.compile(r"(?:\+91[-\s]?)?[6-9]\d{9}\b"),
    "PAN_LIKE_ID": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
}


def redact_pii(text: str) -> tuple[str, dict]:
    """Replace any matched PII with a labelled placeholder.
    Returns the redacted text AND a count of what was redacted, so the
    build script can report what it found without ever printing the
    actual PII to a terminal or log file."""
    redaction_counts = {}
    redacted_text = text
    for label, pattern in PATTERNS.items():
        matches = pattern.findall(redacted_text)
        if matches:
            redaction_counts[label] = len(matches)
            redacted_text = pattern.sub(f"[REDACTED_{label}]", redacted_text)
    return redacted_text, redaction_counts
