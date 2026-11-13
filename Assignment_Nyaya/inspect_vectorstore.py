"""
inspect_vectorstore.py

Disha — Peek Inside the Vector Store
=======================================
Same idea as the Saathi lab: demystify what build_vectorstore.py actually
created in ./chroma_store.

  PART 1 — through the ChromaDB API, showing the collection and a sample
  of stored chunks with their metadata (source, chunk_index, and this
  time also which chunking_strategy produced them).

  PART 2 — directly inside chroma.sqlite3 using plain Python sqlite3,
  listing whatever real tables exist and previewing a few rows.

Run this AFTER build_vectorstore.py:
    python inspect_vectorstore.py
"""
import sqlite3
import chromadb

print("=" * 64)
print("PART 1 — Through the ChromaDB API")
print("=" * 64)

client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_collection("disha_documents")

print(f"\nCollection name : {collection.name}")
print(f"Total chunks    : {collection.count()}")

print("\nA sample of 5 stored chunks, with their metadata:")
sample = collection.peek(limit=5)
for i, (doc, meta) in enumerate(zip(sample["documents"], sample["metadatas"]), start=1):
    print(f"\n  --- Chunk {i} ---")
    print(f"  metadata : {meta}")
    print(f"  text     : {doc[:100]}...")

print("\nNotice 'chunking_strategy' in the metadata — this tells you which")
print("of the 5 strategies (fixed / overlap / sentence / heading_aware /")
print("semantic) produced this specific chunk, since build_vectorstore.py")
print("stamps it onto every chunk it creates.")

print("\n" + "=" * 64)
print("PART 2 — Directly inside chroma.sqlite3")
print("=" * 64)

db_path = "./chroma_store/chroma.sqlite3"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
tables = [row[0] for row in cursor.fetchall() if not row[0].startswith("sqlite_")]

print(f"\nReal tables found inside {db_path}:")
for t in tables:
    print(f"  - {t}")

print("\nPreviewing up to 3 rows from each table:")
for t in tables:
    print(f"\n  --- Table: {t} ---")
    try:
        cursor.execute(f"PRAGMA table_info({t});")
        columns = [col[1] for col in cursor.fetchall()]
        print(f"  columns: {columns}")
        cursor.execute(f"SELECT * FROM {t} LIMIT 3;")
        rows = cursor.fetchall()
        for row in rows:
            print(f"  {row}")
        if not rows:
            print("  (empty)")
    except Exception as e:
        print(f"  (could not read this table: {e})")

conn.close()

print("\n" + "=" * 64)
print("KEY TAKEAWAY")
print("=" * 64)
print("""
Chunk text and metadata (source, chunk_index, chunking_strategy) live as
ordinary SQLite rows. The vectors themselves live in a separate index
structure built for fast nearest-neighbour search. When app.py retrieves
chunks, Chroma searches that index, then resolves the matches back to
real text and metadata via the SQLite tables you just previewed.
""")
