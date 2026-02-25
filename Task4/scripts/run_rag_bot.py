#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag import RAGBot, RAGBotConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="REPL для RAG-бота")
    parser.add_argument("--index-dir", default="vector_index", help="Каталог с индексом FAISS")
    parser.add_argument("--backend", default="mock", choices=["mock", "gemini"], help="LLM backend")
    parser.add_argument("--gemini-model", default="gemini-2.5-flash", help="Модель Gemini (если backend=gemini)")
    parser.add_argument("--top-k", type=int, default=3, help="Сколько чанков забирать из индекса")
    parser.add_argument("--min-score", type=float, default=0.34, help="Порог для ответа вместо 'Я не знаю'")
    parser.add_argument("--disable-pre-prompt-guard", action="store_true", help="Отключить pre-prompt защиту")
    parser.add_argument("--enable-post-filter", action="store_true", help="Включить post-filter вредоносных чанков")
    parser.add_argument("--enable-sanitize-chunks", action="store_true", help="Включить очистку системных конструкций в чанках")
    parser.add_argument("--show-context", action="store_true", help="Показать найденные чанки перед ответом")
    parser.add_argument("--show-raw-context", action="store_true", help="Показать сырые чанки до фильтрации")
    parser.add_argument("--save-dialog", help="Файл JSONL для сохранения диалога")
    args = parser.parse_args()

    bot = RAGBot(
        RAGBotConfig(
            index_dir=args.index_dir,
            top_k=args.top_k,
            min_score=args.min_score,
            llm_backend=args.backend,
            gemini_model=args.gemini_model,
            pre_prompt_guard=not args.disable_pre_prompt_guard,
            post_filter=args.enable_post_filter,
            sanitize_chunks=args.enable_sanitize_chunks,
        )
    )

    save_path = Path(args.save_dialog) if args.save_dialog else None

    print("RAG-бот запущен. Для выхода введите 'exit' или 'quit'.")
    while True:
        try:
            query = input("\nВы> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("Выход.")
            break

        result = bot.answer(query)

        if args.show_raw_context:
            print("\n[Сырой контекст]")
            for i, ch in enumerate(result["raw_retrieved"], start=1):
                print(
                    "{0}. score={1:.4f} | {2} | {3}".format(
                        i, ch["score"], ch["source_path"], ch["title"]
                    )
                )

        if args.show_context:
            print("\n[Контекст]")
            for i, ch in enumerate(result["retrieved"], start=1):
                print(
                    "{0}. score={1:.4f} | {2} | {3}".format(
                        i, ch["score"], ch["source_path"], ch["title"]
                    )
                )
            if result.get("safety"):
                print("[Safety] filtered={0}, sanitized={1}".format(
                    len(result["safety"].get("filtered_chunks", [])),
                    len(result["safety"].get("sanitized_chunks", [])),
                ))

        print("\nБот>")
        print(result["answer"])

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with save_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "query": result["query"],
                            "answer": result["answer"],
                            "retrieved": [
                                {
                                    "score": round(ch["score"], 4),
                                    "source_path": ch["source_path"],
                                    "title": ch["title"],
                                }
                                for ch in result["retrieved"]
                            ],
                            "safety": result.get("safety", {}),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


if __name__ == "__main__":
    main()
