"""
build_vectorstore.py

Disha — Build the Knowledge Base (Step 1 of 3)
=================================================
The "ingest -> chunk -> embed -> index" half of the RAG architecture.

Loads all 5 source documents (PDF, Word, HTML, CSV, JSON) from
documents/, redacts obvious PII, chunks each document using a
configurable strategy, embeds every chunk with OpenAI's embedding model,
and stores everything in a persistent ChromaDB vector store via
LangChain.

Run:
    python build_vectorstore.py                     # default strategy: overlap
    python build_vectorstore.py --strategy fixed
    python build_vectorstore.py --strategy sentence
    python build_vectorstore.py --strategy heading_aware
    python build_vectorstore.py --strategy semantic

Re-run any time you edit a document, or to compare how a different
chunking strategy changes retrieval quality — this always rebuilds the
whole store from scratch.
"""
import argparse
import glob
import os
import shutil

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from loaders import load_document
from chunking import chunk_text, STRATEGIES
from pii import redact_pii

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"  # modern successor to Ada-002; see README
COLLECTION_NAME = "disha_documents"
PERSIST_DIR = "./chroma_store"


def build(strategy: str):
    print("=" * 64)
    print(f"Disha — Building the Knowledge Base (chunking strategy: {strategy})")
    print("=" * 64)

    filepaths = sorted(glob.glob("documents/*"))
    if not filepaths:
        print("No documents found in documents/ — nothing to build.")
        return

    print(f"\nFound {len(filepaths)} source document(s):")
    for f in filepaths:
        print(f"  - {os.path.basename(f)}")

    all_documents = []
    total_pii_redactions = {}

    for filepath in filepaths:
        filename = os.path.basename(filepath)
        print(f"\nProcessing {filename} ...")

        raw_units = load_document(filepath)
        print(f"  Loaded as {len(raw_units)} source unit(s) (page / row / section / record)")

        file_chunk_count = 0
        for unit in raw_units:
            redacted_text, pii_counts = redact_pii(unit["text"])
            for label, count in pii_counts.items():
                total_pii_redactions[label] = total_pii_redactions.get(label, 0) + count

            chunks = chunk_text(redacted_text, strategy=strategy)
            for i, chunk in enumerate(chunks):
                metadata = dict(unit["metadata"])
                metadata["chunk_index"] = i
                metadata["chunking_strategy"] = strategy
                all_documents.append(Document(page_content=chunk, metadata=metadata))
                file_chunk_count += 1

        print(f"  -> {file_chunk_count} chunk(s) from this file")

    print(f"\nTotal chunks to embed: {len(all_documents)}")
    if total_pii_redactions:
        print(f"PII redacted during ingestion: {total_pii_redactions}")
    else:
        print("No PII patterns matched during ingestion.")

    print("\nEmbedding and storing in ChromaDB (this calls the OpenAI API)...")
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

    # Fresh start every run, so re-running after an edit doesn't duplicate data.
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

    Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )

    print("\n" + "=" * 64)
    print(f"Done. {len(all_documents)} chunks stored in {PERSIST_DIR}")
    print(f"Chunking strategy used: {strategy}")
    print("Next step: start app.py (see README.md)")
    print("=" * 64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="overlap", choices=list(STRATEGIES.keys()))
    args = parser.parse_args()
    build(args.strategy)
