import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer


RUS_STOPWORDS = {
    "и", "в", "во", "на", "по", "с", "со", "к", "ко", "о", "об", "от", "до", "из", "за",
    "не", "но", "что", "как", "кто", "где", "когда", "почему", "какой", "какая", "какие",
    "ли", "а", "для", "или", "это", "этот", "эта", "эти", "у", "я", "ты", "он", "она",
    "мы", "они", "его", "ее", "их", "был", "была", "были", "быть", "над", "под", "при",
}


@dataclass
class RAGBotConfig:
    index_dir: str = "vector_index"
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    top_k: int = 3
    min_score: float = 0.34
    llm_backend: str = "mock"  # mock | gemini
    gemini_model: str = "gemini-2.5-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"


class RAGBot:
    def __init__(self, config: Optional[RAGBotConfig] = None) -> None:
        self.config = config or RAGBotConfig()
        self.index_dir = Path(self.config.index_dir)
        self.index = faiss.read_index(str(self.index_dir / "faiss.index"))
        self.chunks = self._load_chunks(self.index_dir / "chunks.jsonl")
        self.embedder = SentenceTransformer(self.config.model_name)
        self.few_shot_examples = self._build_few_shot_examples()

    @staticmethod
    def _load_dotenv(dotenv_path: Optional[Path] = None) -> None:
        path = dotenv_path or Path(".env")
        if not path.exists():
            return
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return

        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

    @staticmethod
    def _load_chunks(path: Path) -> List[Dict]:
        chunks: List[Dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        if not chunks:
            raise ValueError("chunks.jsonl пустой")
        return chunks

    def _build_few_shot_examples(self) -> List[Dict[str, str]]:
        return [
            {
                "q": "Где можно уничтожить Нулевую Печать?",
                "a": (
                    "Шаги:\n"
                    "1. Ищу в базе упоминание места уничтожения Нулевой Печати.\n"
                    "2. В документе про Пепельный Пик сказано, что это единственное место для уничтожения.\n"
                    "Ответ: Нулевую Печать можно уничтожить на Пепельном Пике."
                ),
            },
            {
                "q": "Какую роль в войне играл Железный Шпиль?",
                "a": (
                    "Шаги:\n"
                    "1. Ищу документ, где описан Железный Шпиль.\n"
                    "2. В найденном фрагменте указано, что он был военной базой Хелриона.\n"
                    "3. Там же сказано, что после атаки энтов он был выведен из строя.\n"
                    "Ответ: Железный Шпиль служил военной базой Хелриона и важным центром операций в войне."
                ),
            },
        ]

    def embed_query(self, query: str) -> np.ndarray:
        vec = self.embedder.encode([query], normalize_embeddings=True)
        return vec.astype("float32")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        top_k = top_k or self.config.top_k
        query_vec = self.embed_query(query)
        scores, ids = self.index.search(query_vec, top_k)
        results = []
        for chunk_idx, score in zip(ids[0], scores[0]):
            if chunk_idx < 0:
                continue
            chunk = self.chunks[int(chunk_idx)]
            item = dict(chunk)
            item["score"] = float(score)
            results.append(item)
        return results

    def build_prompt(self, query: str, retrieved_chunks: List[Dict]) -> Dict[str, str]:
        system_prompt = (
            "Ты RAG-помощник по внутренней базе знаний. "
            "Отвечай только на основе переданных фрагментов контекста. "
            "Если данных недостаточно, отвечай строго: 'Я не знаю'. "
            "Сначала покажи краткие шаги рассуждения (CoT в явном виде), затем дай ответ. "
            "В конце перечисли источники."
        )

        few_shot_block = []
        for ex in self.few_shot_examples:
            few_shot_block.append("Q: {0}\nA: {1}".format(ex["q"], ex["a"]))
        few_shot_text = "\n\n".join(few_shot_block)

        context_parts = []
        for i, ch in enumerate(retrieved_chunks, start=1):
            context_parts.append(
                "[Фрагмент {0}] score={1:.4f}\nИсточник: {2}\nЗаголовок: {3}\nТекст: {4}".format(
                    i, ch.get("score", 0.0), ch["source_path"], ch["title"], ch["text"]
                )
            )
        context_text = "\n\n".join(context_parts) if context_parts else "Контекст не найден."

        user_prompt = (
            "Ниже примеры формата ответа (few-shot):\n\n"
            "{few_shot}\n\n"
            "Контекст из базы знаний:\n"
            "{context}\n\n"
            "Вопрос пользователя:\n"
            "{query}\n\n"
            "Ответ оформи так:\n"
            "Шаги:\n"
            "1. ...\n"
            "2. ...\n"
            "Ответ: ...\n"
            "Источники:\n"
            "- ...\n"
        ).format(few_shot=few_shot_text, context=context_text, query=query)

        return {"system": system_prompt, "user": user_prompt}

    def answer(self, query: str) -> Dict:
        retrieved = self.retrieve(query, self.config.top_k)
        prompt = self.build_prompt(query, retrieved)

        if self.config.llm_backend == "gemini":
            text = self._answer_gemini(prompt)
        else:
            text = self._answer_mock(query, retrieved)

        return {
            "query": query,
            "retrieved": retrieved,
            "prompt": prompt,
            "answer": text,
        }

    def _answer_gemini(self, prompt: Dict[str, str]) -> str:
        self._load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "Я не знаю"

        base_url = os.getenv("GEMINI_BASE_URL", self.config.gemini_base_url)
        model_name = os.getenv("GEMINI_MODEL", self.config.gemini_model)
        url = "{0}/models/{1}:generateContent".format(base_url.rstrip("/"), model_name)
        payload = {
            "systemInstruction": {"parts": [{"text": prompt["system"]}]},
            "contents": [{"parts": [{"text": prompt["user"]}]}],
            "generationConfig": {"temperature": 0.2},
        }
        resp = requests.post(
            url,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            return "Я не знаю"
        parts = (((candidates[0] or {}).get("content") or {}).get("parts")) or []
        texts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")]
        return "\n".join(texts).strip() or "Я не знаю"

    def _answer_mock(self, query: str, retrieved: List[Dict]) -> str:
        if not retrieved:
            return "Я не знаю"

        top = retrieved[0]
        if top["score"] < self.config.min_score:
            return "Я не знаю"

        selected_sentences = self._select_supporting_sentences(query, retrieved)
        if not selected_sentences:
            return "Я не знаю"

        sources = []
        for ch in retrieved[:3]:
            src = "{0} ({1})".format(ch["source_path"], ch["title"])
            if src not in sources:
                sources.append(src)

        steps = [
            "Сначала ищу релевантные фрагменты в векторном индексе FAISS.",
            "Затем отбираю предложения с максимальным смысловым совпадением по вопросу.",
            "Формирую ответ только по найденным фрагментам.",
        ]

        answer_line = self._compose_answer_line(query, selected_sentences)
        if answer_line is None:
            return "Я не знаю"

        parts = ["Шаги:"]
        for i, step in enumerate(steps, start=1):
            parts.append("{0}. {1}".format(i, step))
        parts.append("Ответ: {0}".format(answer_line))
        parts.append("Источники:")
        for src in sources:
            parts.append("- {0}".format(src))
        return "\n".join(parts)

    def _compose_answer_line(self, query: str, sentences: List[str]) -> Optional[str]:
        normalized_query = query.lower()
        joined = " ".join(sentences).strip()
        if not joined:
            return None

        if "кто" in normalized_query:
            return joined
        if "где" in normalized_query:
            return joined
        if "почему" in normalized_query or "зачем" in normalized_query:
            return joined
        if "какую роль" in normalized_query or "роль" in normalized_query:
            return joined
        return joined

    def _select_supporting_sentences(self, query: str, retrieved: List[Dict]) -> List[str]:
        query_terms = self._extract_terms(query)
        candidates = []
        for ch in retrieved:
            for sent in self._split_sentences(ch["text"]):
                score = self._sentence_score(sent, query_terms) + ch["score"] * 0.5
                candidates.append((score, sent.strip()))
        candidates.sort(key=lambda x: x[0], reverse=True)

        picked: List[str] = []
        seen = set()
        for score, sent in candidates:
            if score <= 0.05:
                continue
            key = sent.lower()
            if key in seen:
                continue
            picked.append(sent)
            seen.add(key)
            if len(picked) >= 2:
                break
        return picked

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        text = text.replace("\n", " ").strip()
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p for p in parts if p.strip()]

    @staticmethod
    def _extract_terms(text: str) -> List[str]:
        words = re.findall(r"[А-Яа-яA-Za-z0-9-]{2,}", text.lower())
        return [w for w in words if w not in RUS_STOPWORDS]

    def _sentence_score(self, sentence: str, query_terms: List[str]) -> float:
        if not query_terms:
            return 0.0
        sent_lower = sentence.lower()
        hits = 0
        for term in query_terms:
            if term in sent_lower:
                hits += 1
        return hits / max(1, len(query_terms))
