# Disha — the NCS Telco+ Enterprise Document Q&A Pipeline

*A hands-on RAG pipeline lab covering ingestion, chunking, retrieval, context
assembly, and grounded citation — built on the same pattern as Saathi (HR).*

python 
---

## 1. What Is This, in Plain Language

Saathi answered HR questions from 5 plain-text policy files. **Disha** goes
one step further: it ingests **5 real document formats** (PDF, Word, HTML,
CSV, JSON), lets you **compare 5 different chunking strategies**, retrieve
with **3 different search patterns**, and refuses to answer — honestly
saying "I don't know" — when nothing in the knowledge base is a confident
enough match.

This is the full RAG architecture, end to end:

```
ingest -> chunk -> embed -> index -> retrieve -> rerank/dedupe -> answer -> cite
```

Every one of those eight words maps to a real, runnable piece of code in
this repo.

## 2. The Five Documents, Five Formats

| # | File | Format | Loader used |
|---|------|--------|-------------|
| 1 | `01_5g_deployment_sop.pdf` | PDF | PyMuPDF |
| 2 | `02_enterprise_sla.docx` | Word | python-docx |
| 3 | `03_network_ops_faq.html` | HTML | BeautifulSoup |
| 4 | `04_noc_incident_log.csv` | CSV | built-in `csv` module |
| 5 | `05_vendor_equipment_specs.json` | JSON | built-in `json` module |

All fictional documents written for this training lab. Document 3 (the FAQ)
contains a deliberately planted email address and phone number, specifically
so you can watch the PII redaction step catch and mask them during ingestion.

## 3. Architecture — the Full Flow

```
STEP 1 -- build_vectorstore.py  (run once, or re-run per experiment)

    documents/*.{pdf,docx,html,csv,json}
        |
        |  loaders.py    -- one loader per format, same output shape
        v
    raw text units (one per page / row / section / record)
        |
        |  pii.py        -- redact emails, phone numbers, PAN-style IDs
        v
    redacted text
        |
        |  chunking.py   -- pick ONE of 5 strategies:
        |                   fixed | overlap | sentence | heading_aware | semantic
        v
    chunks, each tagged with source + chunk_index + chunking_strategy
        |
        |  LangChain + OpenAIEmbeddings  -- embed every chunk
        v
    ./chroma_store  (ChromaDB vector store, persisted to disk)


STEP 2 -- app.py  (FastAPI backend, keep running in Terminal 1)

    /ask?q=...&pattern=topk|mmr|threshold&k=3

    1. retrieval.py     -- run the chosen retrieval pattern
    2. confidence check -- if nothing clears MIN_CONFIDENCE, answer
                            "I don't know" and STOP (no LLM call at all)
    3. context.py        -- deduplicate near-identical chunks, then pack
                            into a character budget (evidence packing)
    4. LLM call          -- system prompt requires a (Source: filename)
                            citation on every factual claim
    5. return JSON        -- {retrieved_chunks, answer, sources}


STEP 3 -- agent.py  (run in Terminal 2, while app.py keeps running)

    Sends 5 hardcoded questions -- one per document format -- to /ask,
    then a BONUS section comparing topk vs. mmr vs. threshold on one
    question, so the retrieval-pattern differences are visible directly.


STEP 4 (optional) -- inspect_vectorstore.py

    Peeks inside ./chroma_store -- through the ChromaDB API, then
    directly into the underlying chroma.sqlite3 file -- to show exactly
    where chunk text and metadata physically live on disk.
```

## 4. Setup — Step by Step

### Step 1 — Clone the repo
```
git clone <the link shared with you in the lab>
cd disha-rag-pipeline
```

### Step 2 — Create your `.env` file
Copy `.env.example` to a new file named exactly `.env`:
```
OPENAI_API_KEY="paste-the-key-shared-in-the-lab-here"
```
**Never share this file or commit it to Git** — it's already in `.gitignore`.

### Step 3 — Set up your environment and install dependencies
```
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Step 4 — Build the vector store (run once)
```
python build_vectorstore.py
```
By default this uses the **overlap** chunking strategy. To compare a
different one:
```
python build_vectorstore.py --strategy heading_aware
python build_vectorstore.py --strategy sentence
python build_vectorstore.py --strategy fixed
python build_vectorstore.py --strategy semantic
```
Each run reports how many chunks it created, and what PII (if any) it
redacted along the way. Re-run this any time you change a document, or want
to try a different strategy — it always rebuilds `./chroma_store` from
scratch.

### Step 5 — Start Disha (Terminal 1)
```
uvicorn app:app --reload --port 8000
```
Leave this running. Visit `http://localhost:8000/docs` any time to try
questions interactively from a browser.

### Step 6 — Ask Disha your questions (Terminal 2)
```
python agent.py
```
You'll see all 5 hardcoded questions (one per document format), the
retrieved chunk metadata behind each answer, the grounded answer itself,
and a bonus section comparing all 3 retrieval patterns on one question.

## 5. The 5 Hardcoded Questions in `agent.py`

1. What is the maximum indoor EIRP for Band 78 deployments? → *PDF (SOP)*
2. What service credit applies if uptime falls below the SLA guarantee? → *Word (SLA)*
3. Who do I contact for a P1 escalation outside business hours? → *HTML (FAQ) — also tests PII redaction*
4. Which incidents in the log are still Open? → *CSV (incident log)*
5. Which vendor's equipment is NOT approved for indoor deployment, and why? → *JSON (equipment specs)*

## 6. What to Notice While Running This

- **Multi-format grounding**: all 5 answers cite a real filename — a PDF, a
  Word doc, an HTML page, a CSV row, and a JSON record — using the exact
  same pipeline code.
- **PII redaction**: run `build_vectorstore.py` and read its console output
  — it should report redacting 1 email and 1 phone number from the HTML FAQ.
  Ask agent's Q3 and check the answer never reveals the raw email/phone,
  only that a contact exists.
- **"I don't know" in action**: ask something no document covers (e.g., via
  `curl "http://localhost:8000/ask?q=What+is+our+corporate+tax+rate%3F"`) —
  Disha should refuse rather than guess, and `retrieved_chunks` should be empty.
- **Chunking strategy differences**: rebuild the store with `--strategy
  heading_aware` versus `--strategy fixed`, then re-run `agent.py`'s Q2
  (the SLA question). `heading_aware` should retrieve one clean, complete
  clause; `fixed` may retrieve a chunk that cuts a clause in half.
- **Retrieval pattern differences**: the BONUS section at the end of
  `agent.py`'s output runs the same question through `topk`, `mmr`, and
  `threshold` — compare how many chunks each returns and whether the answer
  changes.

## 7. A Note on Retrieval Confidence Scores

`MIN_CONFIDENCE = 0.3` in `app.py` is a starting point, not a universal
constant — the "right" threshold depends on your embedding model and your
documents. If Disha says "I don't know" to questions you'd expect it to
answer, try lowering this value; if it answers questions it shouldn't be
confident about, raise it. This is exactly the kind of number worth tuning
live with the cohort, not treating as fixed.

## 8. A Note on the Model Names

`CHAT_MODEL` (`gpt-5.4-mini`) and `EMBED_MODEL` (`text-embedding-3-small`)
are current models as of this lab's delivery date. The official design-doc
tool list references `OpenAI Ada-002` for embeddings — that model has since
been superseded by the `text-embedding-3` family, which is what this repo
uses. Substitute whichever models your organisation has approved.

## 9. Troubleshooting

| Problem | Likely cause |
|---|---|
| `ModuleNotFoundError` | Virtual environment not activated, or `pip install -r requirements.txt` not run |
| `AuthenticationError` | Check `.env` has your real key, correctly quoted |
| `build_vectorstore.py` fails on the PDF | Confirm `pymupdf` installed correctly — some environments need `pip install pymupdf` run separately |
| `app.py` fails with a "collection not found" error | Run `build_vectorstore.py` first, from this same folder |
| `agent.py` says "Could not reach Disha" | `app.py` isn't running in the other terminal |
| Every answer is "I don't know" | `MIN_CONFIDENCE` may be set too high for your embedding model — see Section 7 |

## 10. Sample Assignment Outline (for Participants)

Use this as a graded or self-check exercise after completing the guided
walkthrough above.

**Assignment: Extend Disha with a 6th Document**

1. Add ONE new document to `documents/` in any format already supported
   (PDF, Word, HTML, CSV, or JSON). Suggested topic: a "Change Management
   Policy" — what approvals are needed before a network change, and the
   standard notice period.
2. If your new format needs no new loader, skip to step 3. If you'd like
   an extra challenge, add support for a 6th format not yet covered (e.g.,
   Markdown or plain `.txt`) by writing a new function in `loaders.py` and
   registering it in the `LOADERS` dispatch table.
3. Re-run `build_vectorstore.py` and confirm the console output shows your
   new document being processed and chunked.
4. Add a 6th question to `agent.py`'s `QUESTIONS` list, targeting a fact
   that ONLY exists in your new document.
5. Run `agent.py` and confirm: (a) the new question is answered correctly,
   citing your new document by name, and (b) all 5 original questions still
   answer correctly — a good pipeline shouldn't regress on old data when
   new data is added.
6. **Written reflection (150–200 words):** Which chunking strategy did you
   choose for your new document, and why did it fit that document's
   structure better than the alternatives? What would you check first if
   your new question's answer cited the wrong source?

**Definition of done:**
- 6 documents ingested without errors
- `agent.py` runs all 6 questions successfully with correct citations
- A short written reflection answering the two questions in step 6

## 11. Where This Goes Next

This lab covers the full single-pass RAG architecture — one retrieval, one
LLM call, no memory across questions, no re-ranking model, no evaluation
harness. Later modules add: cross-encoder re-ranking on top of these
retrieval patterns, automated evaluation (Ragas) against a curated test
set, and eventually multi-agent orchestration where a retrieval step like
this one becomes just one tool inside a larger agent pipeline.
