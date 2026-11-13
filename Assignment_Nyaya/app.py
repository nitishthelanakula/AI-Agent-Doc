"""
app.py

Disha — the NCS Telco+ Enterprise Document Q&A Pipeline (Step 2 of 3)
=========================================================================
Implements the rest of the RAG architecture: retrieve -> assemble context
(dedupe + pack) -> answer -> cite (or say "I don't know").

Run AFTER build_vectorstore.py has been run at least once:
    uvicorn app:app --reload --port 8000

Try a question directly in a browser once running:
    http://localhost:8000/ask?q=What+is+the+maximum+indoor+EIRP+for+Band+78%3F
"""
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI

from retrieval import retrieve
from context import assemble_context

load_dotenv()

CHAT_MODEL = "gpt-5.4-mini"             # substitute your organisation's approved model
EMBED_MODEL = "text-embedding-3-small"   # must match build_vectorstore.py
COLLECTION_NAME = "disha_documents"
PERSIST_DIR = "./chroma_store"
MIN_CONFIDENCE = 0.3   # below this relevance score, we refuse to answer rather than guess

SYSTEM_PROMPT = """
You are Disha, the NCS Telco+ Enterprise Document Q&A Assistant.

Tone: Precise, calm, and technical — the voice of a senior engineer.

Grounding rule: Only answer using the excerpts provided in each question.
Every factual claim in your answer MUST be followed by a citation in the
form (Source: <filename>). If the excerpts don't cover the question, say
so plainly and do not guess.
"""

client = OpenAI()
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR,
)

app = FastAPI(title="Disha - NCS Telco+ Enterprise Document Q&A Pipeline")


@app.get("/")
def root():
    return {"message": "Disha is running. Try /ask?q=your+question&pattern=topk"}


@app.get("/ask")
def ask(q: str, pattern: str = "topk", k: int = 3):
    """
    pattern: "topk" (default), "mmr", or "threshold" — see retrieval.py
    k: how many chunks to retrieve
    """
    results = retrieve(vectorstore, q, pattern=pattern, k=k)

    # "I don't know" guardrail. If nothing came back, or the best available
    # score is below our confidence floor, refuse to answer rather than
    # asking the LLM to generate something from weak or no evidence.
    scored = [(doc, score) for doc, score in results if score is not None]
    best_score = max((score for _, score in scored), default=None)
    if not results or (best_score is not None and best_score < MIN_CONFIDENCE):
        return {
            "question": q,
            "pattern": pattern,
            "retrieved_chunks": [],
            "answer": ("I don't know \u2014 none of the ingested documents cover this "
                       "question confidently enough for me to answer."),
            "sources": [],
        }

    context_text, used_results = assemble_context(results)

    retrieved_chunks = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "metadata": {k2: v for k2, v in doc.metadata.items() if k2 != "source"},
            "relevance_score": round(score, 4) if score is not None else None,
            "text_preview": doc.page_content[:150] + ("..." if len(doc.page_content) > 150 else ""),
        }
        for doc, score in used_results
    ]

    user_prompt = (
        f"Question: {q}\n\n"
        f"Retrieved excerpts:\n{context_text}\n\n"
        "Answer using only the excerpts above. Cite each fact as (Source: <filename>)."
    )

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    return {
        "question": q,
        "pattern": pattern,
        "retrieved_chunks": retrieved_chunks,
        "answer": response.choices[0].message.content,
        "sources": sorted(set(doc.metadata.get("source", "unknown") for doc, _ in used_results)),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
