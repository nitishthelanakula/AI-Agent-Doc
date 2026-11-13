"""
loaders.py

Disha — Multi-Format Document Loaders
=======================================
Each function below takes a file path and returns a list of plain Python
dicts: {"text": <string>, "metadata": {"source": <filename>, ...}}.

Why one dict shape for every format? Because everything downstream
(chunking, embedding, storage) only needs to know "here is some text, and
here is where it came from" — it doesn't care whether that text originally
lived in a PDF, a Word doc, an HTML page, a CSV row, or a JSON object.
That's the whole point of a document loader layer.

Kept deliberately simple and explainable:
  - PDF   -> PyMuPDF (fitz)       — extracts text page by page
  - Word  -> python-docx          — extracts text paragraph by paragraph
  - HTML  -> BeautifulSoup        — strips tags, keeps heading + paragraph text
  - CSV   -> built-in csv module  — one "document" per row
  - JSON  -> built-in json module — one "document" per top-level record

A note on "Unstructured": the official design-doc tool list also mentions
the `unstructured` library, which can auto-detect and parse many formats
through a single interface. We use per-format libraries here instead,
specifically so each loader is short enough to read top-to-bottom in a
live session — `unstructured` is worth exploring afterward as a more
robust, production-grade alternative that handles messier real-world files.
"""
import csv
import json
import fitz  # PyMuPDF
import docx  # python-docx
from bs4 import BeautifulSoup


def load_pdf(filepath: str) -> list[dict]:
    """One 'document' per page — keeps citations pointing at a specific page."""
    docs = []
    pdf = fitz.open(filepath)
    for page_num, page in enumerate(pdf, start=1):
        text = page.get_text().strip()
        if text:
            docs.append({
                "text": text,
                "metadata": {"source": filepath.split("/")[-1], "page": page_num},
            })
    pdf.close()
    return docs


def load_docx(filepath: str) -> list[dict]:
    """One 'document' for the whole file — Word docs here are short enough
    that page-level splitting isn't meaningful; chunking handles the rest."""
    d = docx.Document(filepath)
    full_text = "\n".join(p.text for p in d.paragraphs if p.text.strip())
    return [{
        "text": full_text,
        "metadata": {"source": filepath.split("/")[-1]},
    }]


def load_html(filepath: str) -> list[dict]:
    """One 'document' per <h2> section — HTML FAQs are naturally structured
    this way, so splitting at headings keeps each Q&A pair intact."""
    with open(filepath, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    docs = []
    filename = filepath.split("/")[-1]

    # Grab the intro paragraph (document id/version line) as its own chunk.
    intro = soup.find("p")
    if intro:
        docs.append({"text": intro.get_text(strip=True),
                      "metadata": {"source": filename, "section": "header"}})

    for heading in soup.find_all("h2"):
        section_title = heading.get_text(strip=True)
        paragraph = heading.find_next_sibling("p")
        section_text = paragraph.get_text(strip=True) if paragraph else ""
        docs.append({
            "text": f"{section_title}\n{section_text}",
            "metadata": {"source": filename, "section": section_title},
        })
    return docs


def load_csv(filepath: str) -> list[dict]:
    """One 'document' per row — natural for tabular incident/ticket logs."""
    docs = []
    filename = filepath.split("/")[-1]
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = ", ".join(f"{key}: {value}" for key, value in row.items())
            docs.append({
                "text": text,
                "metadata": {"source": filename, "row_id": row.get("incident_id", "")},
            })
    return docs


def load_json(filepath: str) -> list[dict]:
    """One 'document' per record in the top-level list this file contains."""
    filename = filepath.split("/")[-1]
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    records = data.get("equipment", [])  # this file's top-level list key
    for record in records:
        text = ", ".join(f"{key}: {value}" for key, value in record.items())
        docs.append({
            "text": text,
            "metadata": {"source": filename, "vendor": record.get("vendor", "")},
        })
    return docs


# Dispatch table: pick the right loader based on file extension.
LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".html": load_html,
    ".csv": load_csv,
    ".json": load_json,
}


def load_document(filepath: str) -> list[dict]:
    """Single entry point — figures out which loader to use from the extension."""
    ext = "." + filepath.rsplit(".", 1)[-1].lower()
    if ext not in LOADERS:
        raise ValueError(f"No loader registered for file type: {ext}")
    return LOADERS[ext](filepath)
