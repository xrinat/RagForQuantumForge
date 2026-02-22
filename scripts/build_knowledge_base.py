#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


def load_terms_map(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        raise ValueError("terms_map.json должен содержать непустой JSON-объект")
    return data


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def replace_terms(text: str, terms_map: Dict[str, str]) -> Tuple[str, int]:
    replacements = 0
    sorted_items = sorted(terms_map.items(), key=lambda x: len(x[0]), reverse=True)
    for source, target in sorted_items:
        pattern = re.compile(rf"(?<!\w){re.escape(source)}(?!\w)")
        text, count = pattern.subn(target, text)
        replacements += count
    return text, replacements


def find_leftovers(text: str, terms_map: Dict[str, str]) -> List[str]:
    leftovers = []
    for source in terms_map:
        pattern = re.compile(rf"(?<!\w){re.escape(source)}(?!\w)")
        if pattern.search(text):
            leftovers.append(source)
    return leftovers


def build(raw_dir: Path, out_dir: Path, terms_map: Dict[str, str], min_docs: int) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_files = sorted(raw_dir.glob("*.md"))
    if len(raw_files) < min_docs:
        raise ValueError(f"Expected at least {min_docs} documents in {raw_dir}, got {len(raw_files)}")

    report_docs = []
    total_replacements = 0
    failed_docs = 0

    for src_file in raw_files:
        raw_text = src_file.read_text(encoding="utf-8")
        cleaned = clean_text(raw_text)
        transformed, replacements = replace_terms(cleaned, terms_map)
        leftovers = find_leftovers(transformed, terms_map)

        out_file = out_dir / src_file.name
        out_file.write_text(transformed, encoding="utf-8")

        total_replacements += replacements
        if leftovers:
            failed_docs += 1
        report_docs.append(
            {
                "file": src_file.name,
                "replacements": replacements,
                "leftover_terms_count": len(leftovers),
                "leftover_terms_sample": leftovers[:10],
            }
        )

    report = {
        "documents_processed": len(raw_files),
        "total_replacements": total_replacements,
        "documents_with_leftovers": failed_docs,
        "documents": report_docs,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Собрать финальную базу знаний из сырых LOTR-источников")
    parser.add_argument("--raw-dir", default="rawSources", help="Каталог с исходными markdown-файлами")
    parser.add_argument("--out-dir", default="knowledge_base", help="Каталог для преобразованных файлов")
    parser.add_argument("--terms-map", default="terms_map.json", help="Путь к JSON-словарю подмены терминов")
    parser.add_argument("--min-docs", type=int, default=30, help="Минимальное число исходных документов")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    terms_map_path = Path(args.terms_map)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Каталог исходников не найден: {raw_dir}")
    if not terms_map_path.exists():
        raise FileNotFoundError(f"Файл словаря не найден: {terms_map_path}")

    terms_map = load_terms_map(terms_map_path)
    report = build(raw_dir, out_dir, terms_map, args.min_docs)

    report_path = out_dir / "build_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Готово. Обработано документов: {report['documents_processed']}.")
    print(f"Всего замен: {report['total_replacements']}")
    print(f"Документов с остатками терминов: {report['documents_with_leftovers']}")
    print(f"Отчет: {report_path}")


if __name__ == "__main__":
    main()
