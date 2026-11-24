import json
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = BASE_DIR / "data" / "index"
MODEL_NAME = "BAAI/bge-small-en-v1.5"


class LocalRetriever:
    def __init__(self) -> None:
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index(str(INDEX_DIR / "cs.faiss"))
        self.chunks = json.loads(
            (INDEX_DIR / "chunks.json").read_text(encoding="utf-8")
        )

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        # BGE retrieval prefix improves query-to-passage matching
        formatted = f"Represent this sentence for searching relevant passages: {query}"
        q_vec = self.model.encode([formatted], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_vec)

        scores, indices = self.index.search(q_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({**chunk, "score": round(float(score), 4)})
        return results


if __name__ == "__main__":
    retriever = LocalRetriever()
    while True:
        query = input("\nTest CS query (or 'exit'): ").strip()
        if not query or query.lower() == "exit":
            break
        for hit in retriever.search(query, top_k=5):
            print(f"\n[{hit['chunk_id']} | {hit['source']} p.{hit['page']} | score={hit['score']}]")
            print(hit["text"][:400], "...")