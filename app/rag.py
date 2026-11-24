from pathlib import Path

import chromadb

from app.bedrock_client import embed_text, generate_answer

CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"

_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
_collection = _client.get_or_create_collection("docs")

PROMPT_TEMPLATE = """Answer the question using only the context below. \
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""


def answer_question(question: str, top_k: int = 3) -> dict:
    query_embedding = embed_text(question)
    results = _collection.query(query_embeddings=[query_embedding], n_results=top_k)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context = "\n\n".join(documents)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    answer = generate_answer(prompt)

    sources = sorted({m["source"] for m in metadatas})
    return {"answer": answer, "sources": sources}
