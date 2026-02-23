#!/usr/bin/env python3
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag import RAGBot, RAGBotConfig


SUCCESS_QUERIES = [
    "Кто нес Нулевую Печать к Пепельному Пику?",
    "Какую роль играл Железный Шпиль в войне?",
    "Почему Белый Бастион был важен для Валенара?",
    "Кто помог разрушить Железный Шпиль?",
]

UNKNOWN_QUERIES = [
    "Как зовут столицу планеты Ти'лора?",
    "Какой двигатель использует HyperRelay?",
]


def main() -> None:
    out_path = Path(__file__).resolve().parents[1] / "rag_dialog_examples.json"
    bot = RAGBot(RAGBotConfig(llm_backend="mock"))

    rows = []
    for query in SUCCESS_QUERIES + UNKNOWN_QUERIES:
        result = bot.answer(query)
        rows.append(
            {
                "query": query,
                "answer": result["answer"],
                "top_hits": [
                    {
                        "source_path": ch["source_path"],
                        "title": ch["title"],
                        "score": round(ch["score"], 4),
                    }
                    for ch in result["retrieved"][:3]
                ],
            }
        )

    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Готово. Примеры диалогов сохранены в {0}".format(out_path))


if __name__ == "__main__":
    main()
