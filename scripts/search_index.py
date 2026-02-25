#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

import faiss
from sentence_transformers import SentenceTransformer


def load_chunks(path: Path) -> List[Dict]:
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Поиск по FAISS-индексу")
    parser.add_argument("--query", required=True, help="Поисковый запрос")
    parser.add_argument("--index-dir", default="vector_index", help="Каталог с индексом")
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Модель эмбеддингов",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Количество результатов")
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    index = faiss.read_index(str(index_dir / "faiss.index"))
    chunks = load_chunks(index_dir / "chunks.jsonl")

    model = SentenceTransformer(args.model_name)
    query_vector = model.encode([args.query], normalize_embeddings=True)
    scores, ids = index.search(query_vector.astype("float32"), args.top_k)

    print("Запрос: {0}".format(args.query))
    print("Результаты:")
    for rank, (idx, score) in enumerate(zip(ids[0], scores[0]), start=1):
        if idx < 0:
            continue
        chunk = chunks[int(idx)]
        print("{0}. score={1:.4f} | {2} | {3}".format(rank, float(score), chunk["source_path"], chunk["title"]))
        print("   {0}".format(chunk["text"][:220].replace("\n", " ")))


if __name__ == "__main__":
    main()
