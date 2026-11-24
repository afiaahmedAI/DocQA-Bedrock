from dotenv import load_dotenv

load_dotenv()

from pathlib import Path

import chromadb

from app.bedrock_client import embed_text
from app.chunker import chunk_text

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"


def main():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection("docs")

    doc_files = sorted(DOCS_DIR.glob("*.txt"))
    if not doc_files:
        print(f"No .txt files found in {DOCS_DIR}")
        return

    ids, embeddings, documents, metadatas = [], [], [], []
    chunk_count = 0

    for doc_file in doc_files:
        text = doc_file.read_text()
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            ids.append(f"{doc_file.name}-{i}")
            embeddings.append(embed_text(chunk))
            documents.append(chunk)
            metadatas.append({"source": doc_file.name})
            chunk_count += 1

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"Ingested {chunk_count} chunks from {len(doc_files)} files into {CHROMA_DIR}")


if __name__ == "__main__":
    main()
