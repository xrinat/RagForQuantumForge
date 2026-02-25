#!/usr/bin/env python3
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def read_markdown_docs(kb_dir: Path) -> List[Dict]:
    docs = []
    for path in sorted(kb_dir.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        lines = text.splitlines()
        title = lines[0].replace("#", "").strip() if lines and lines[0].startswith("#") else path.stem
        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else text
        docs.append({"path": str(path).replace("\\", "/"), "title": title, "text": body})
    if not docs:
        raise ValueError("В knowledge_base не найдено markdown-документов для индексации")
    return docs


def chunk_by_words(text: str, chunk_words: int, overlap_words: int) -> List[Tuple[str, int, int]]:
    words = text.split()
    if not words:
        return []
    if chunk_words <= overlap_words:
        raise ValueError("chunk_words должен быть больше overlap_words")

    chunks = []
    step = chunk_words - overlap_words
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunk_text = " ".join(words[start:end]).strip()
        if chunk_text:
            chunks.append((chunk_text, start, end))
        if end == len(words):
            break
        start += step
    return chunks


def build_chunks(docs: List[Dict], chunk_words: int, overlap_words: int) -> List[Dict]:
    chunks = []
    chunk_id = 0
    for doc in docs:
        doc_chunks = chunk_by_words(doc["text"], chunk_words, overlap_words)
        for local_idx, (chunk_text, word_start, word_end) in enumerate(doc_chunks):
            chunks.append(
                {
                    "id": chunk_id,
                    "chunk_id": "chunk-{0:05d}".format(chunk_id),
                    "source_path": doc["path"],
                    "title": doc["title"],
                    "local_chunk_index": local_idx,
                    "word_start": word_start,
                    "word_end": word_end,
                    "text": chunk_text,
                }
            )
            chunk_id += 1
    if not chunks:
        raise ValueError("После чанкинга не получено ни одного чанка")
    return chunks


def encode_texts(model: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vectors.astype("float32")


def save_chunks_jsonl(chunks: List[Dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in chunks:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def run_queries(
    model: SentenceTransformer,
    index: faiss.Index,
    chunks: List[Dict],
    queries: List[str],
    top_k: int,
) -> List[Dict]:
    query_vectors = encode_texts(model, queries, batch_size=16)
    scores, ids = index.search(query_vectors, top_k)
    results = []
    for q_idx, query in enumerate(queries):
        hits = []
        for rank, (chunk_idx, score) in enumerate(zip(ids[q_idx], scores[q_idx]), start=1):
            if chunk_idx < 0:
                continue
            chunk = chunks[int(chunk_idx)]
            hits.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "chunk_id": chunk["chunk_id"],
                    "source_path": chunk["source_path"],
                    "title": chunk["title"],
                    "text_preview": chunk["text"][:240],
                }
            )
        results.append({"query": query, "hits": hits})
    return results


def write_index_readme(
    out_dir: Path,
    model_name: str,
    model_link: str,
    embedding_dim: int,
    knowledge_base_path: str,
    chunks_count: int,
    build_seconds: float,
) -> None:
    content = (
        "# Векторный индекс\n\n"
        "## Параметры\n"
        "- Модель эмбеддингов: `{0}`\n"
        "- Ссылка на модель/API: `{1}`\n"
        "- Размер эмбеддинга: `{2}`\n"
        "- База знаний: `{3}`\n"
        "- Количество чанков: `{4}`\n"
        "- Время генерации (сек): `{5:.2f}`\n\n"
        "## Файлы\n"
        "- `faiss.index`\n"
        "- `chunks.jsonl`\n"
        "- `build_meta.json`\n"
        "- `sample_queries.json`\n"
    ).format(model_name, model_link, embedding_dim, knowledge_base_path, chunks_count, build_seconds)
    (out_dir / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Создание FAISS-индекса по knowledge_base")
    parser.add_argument("--kb-dir", default="knowledge_base", help="Каталог базы знаний")
    parser.add_argument("--out-dir", default="vector_index", help="Каталог для артефактов индекса")
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Название эмбеддинг-модели Sentence Transformers",
    )
    parser.add_argument("--chunk-words", type=int, default=180, help="Размер чанка в словах")
    parser.add_argument("--overlap-words", type=int, default=40, help="Перекрытие чанков в словах")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size для эмбеддингов")
    parser.add_argument("--top-k", type=int, default=3, help="Число результатов в примерах поиска")
    args = parser.parse_args()

    kb_dir = Path(args.kb_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    docs = read_markdown_docs(kb_dir)
    chunks = build_chunks(docs, chunk_words=args.chunk_words, overlap_words=args.overlap_words)

    model = SentenceTransformer(args.model_name)
    vectors = encode_texts(model, [c["text"] for c in chunks], batch_size=args.batch_size)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, str(out_dir / "faiss.index"))
    save_chunks_jsonl(chunks, out_dir / "chunks.jsonl")

    sample_queries = [
        "Кто нес Нулевую Печать к Пепельному Пику?",
        "Какую роль сыграл Железный Шпиль в войне?",
        "Почему Белый Бастион был критически важен для Валенара?",
    ]
    sample_results = run_queries(model, index, chunks, sample_queries, top_k=args.top_k)
    (out_dir / "sample_queries.json").write_text(
        json.dumps(sample_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    duration = time.time() - started
    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "knowledge_base_dir": str(kb_dir).replace("\\", "/"),
        "documents_count": len(docs),
        "chunks_count": len(chunks),
        "model_name": args.model_name,
        "model_link": "https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "embedding_dim": int(dim),
        "chunk_words": args.chunk_words,
        "overlap_words": args.overlap_words,
        "build_seconds": round(duration, 3),
        "index_type": "faiss.IndexFlatIP (cosine similarity on normalized embeddings)",
    }
    (out_dir / "build_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    write_index_readme(
        out_dir=out_dir,
        model_name=args.model_name,
        model_link=meta["model_link"],
        embedding_dim=int(dim),
        knowledge_base_path=str(kb_dir).replace("\\", "/"),
        chunks_count=len(chunks),
        build_seconds=duration,
    )

    print("Готово. Индекс сохранен в: {0}".format(out_dir))
    print("Документов: {0}, чанков: {1}, dim: {2}".format(len(docs), len(chunks), dim))
    print("Время генерации: {0:.2f} сек".format(duration))


if __name__ == "__main__":
    main()
