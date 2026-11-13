# Anvesh Challenge — Starter Kit (unobserved)

*This is the "before" state. Read the companion document,
**Anvesh_Observability_Challenge_Outline.docx**, before writing any code —
it shows the complete target architecture first, names all three
inconsistencies seeded in this repo on purpose, and gives a step-by-step
plan (with suggested GitHub Copilot prompts) to build OpenTelemetry +
Jaeger tracing AND the full metrics/eval/drift/alerting/feedback layer on
top of it.*

## What this is

The Anvesh fraud/AML transaction-pattern investigation agent (same
scenario as the Day-1 repo), extended with one more real step: once the
main tool loop finishes gathering evidence, it **delegates to a second,
separate model call** — a SAR-drafting sub-agent (`src/core/sub_agent.py`)
— to produce the final narrative a human reviewer would read.

Built on the new folder structure (`src/core/`, `src/tools/`, `src/data/`,
`tests/`, `scripts/`) used by every repo in this series from Anvesh
onward.

## What this is NOT (yet)

**Zero observability of any kind.** No OpenTelemetry, no Jaeger, no
Prometheus, no Grafana, no metrics, no online evaluation, no drift
detection, no alerting, no human feedback loop. `docker-compose.yml` does
not exist in this repo at all. Every one of those is something you build,
in the order the companion outline lays out.

## The three inconsistencies — seeded, not described

All three are **already live in the code**, every time you run it — they
are not something you need to "turn on" to encounter them, only something
you can `seed_incident()` on demand to study one at a time in isolation.
None of them raise an error, log a warning, or show up as a failed HTTP
call. All three were written so that **reading a single trace once you've
added tracing will not, by itself, make them obvious** — that gap is the
entire reason the later steps (metrics, eval, drift, alerting) exist.

| # | Name | What's actually wrong | Where it lives |
|---|---|---|---|
| 1 | Tool call issue | `check_aml_typology()` is silently rerouted to always run the `rapid_fund_rotation` rule, no matter which typology was requested — CUST-10587's structuring pattern stops being flagged | `src/core/incidents.py` → `effective_typology()`, called from `src/tools/investigation_tools.py` |
| 2 | Model drift | Every `gpt-4o-mini` request is silently served by `gpt-4o` instead — no error, higher cost, and the caller's own "requested model" value is unaffected | `src/core/incidents.py` → `effective_model()`, called from `src/core/llm_provider.py` |
| 3 | Sub-agent call issue | The SAR-drafting sub-agent delegation drops all of the investigation evidence — it still returns a confident, fluent SAR narrative, just one that reflects nothing that was actually found | `src/core/incidents.py` → `sub_agent_context_enabled()`, called from `src/core/sub_agent.py` |

Use `POST /seed-incident/{1,2,3}`, `POST /clear-incident`, and
`GET /incident-status` (which only ever says `true`/`false`, never which
one) to control these — the same honesty convention used across this
entire training series. `GET /reveal-incident/{n}` is the answer key; look
at it only after trying to diagnose the incident yourself.

## Setup

1. `cp .env.example .env` and fill in `OPENAI_API_KEY`.
2. `python -m venv .venv && source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. Sanity-check the mock data (no API key needed):
   `python scripts/list_mock_customers.py`
5. Run the deterministic tests (no API key needed):
   `pytest tests/unit -v`
6. Run the demo (needs `OPENAI_API_KEY`) — this runs clean, then seeds and
   clears each incident in turn so you can see all three once by eye
   before you have any instrumentation at all: `python agent.py`
7. Or run the API: `uvicorn app:app --reload --port 8000`

## Where to go next

Open **Anvesh_Observability_Challenge_Outline.docx**. Section 1 shows the
complete target architecture. Section 2 names these three inconsistencies
again, in the context of what will eventually catch each one. Section 3 is
the step-by-step build plan — tracing first, then the five files
(`metrics.py`, `online_eval.py`, `drift.py`, `alerting.py`, `feedback.py`)
one at a time, each with a suggested Copilot prompt.

## File map

| Path | Status |
|---|---|
| `src/core/llm_provider.py` | Working — raw model-call primitive, now routed through `incidents.effective_model()` |
| `src/core/agent_loop.py` | Working — tool loop + hand-off to the SAR sub-agent |
| `src/core/sub_agent.py` | Working — second model call drafting the SAR narrative |
| `src/core/incidents.py` | Working — all 3 inconsistencies, seed/clear/reveal |
| `src/core/pricing.py` | Prices both `gpt-4o-mini` and `gpt-4o` |
| `src/tools/investigation_tools.py` | Working — 4 tools, one routed through `incidents.effective_typology()` |
| `src/data/mock_store.py` | Mock customers, transactions, watchlist, typology rules |
| `tests/unit/test_tools.py`, `test_incidents.py` | Deterministic tests, no API key needed |
| `otel_setup.py`, `metrics.py`, `online_eval.py`, `drift.py`, `alerting.py`, `feedback.py` | **Do not exist yet** — these are what you build |
| `docker-compose.yml`, Prometheus/Grafana config | **Do not exist yet** |
