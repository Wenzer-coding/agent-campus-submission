"""语料加载、文本归一化与确定性词法检索。

设计约束（见 docs/02-solution.md §8、§9）：
- 检索为纯函数：同一语料 + 同一 query 永远得到同样的排序。
- 归一化不删除标点，防止引用校验被跨句拼接绕过。
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG = {
    "threshold": 0.2,
    "title_weight": 2.0,
    "top_k_default": 5,
    "top_k_max": 10,
}

_ZERO_WIDTH = "​‌‍‎‏﻿"


def normalize(text: str) -> str:
    """NFKC + casefold + 去零宽字符 + 空白折叠为单空格；不删除标点。"""
    t = unicodedata.normalize("NFKC", text).casefold()
    t = "".join(ch for ch in t if ch not in _ZERO_WIDTH)
    return " ".join(t.split())


def bigrams(text: str) -> set[str]:
    """字符 bigram 集合（检索用）；标点/符号/分隔符不参与，避免问号等噪声畸变评分。

    注意：引用保真校验使用的是 normalize()，那里刻意保留标点，防止跨句拼接绕过校验。
    """
    compact = "".join(
        ch for ch in normalize(text) if unicodedata.category(ch)[0] not in ("P", "S", "Z")
    )
    if len(compact) < 2:
        return set()
    return {compact[i : i + 2] for i in range(len(compact) - 1)}


_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？…!?；;\n])")


def split_sentences(text: str) -> list[str]:
    """整句切分（按 。！？…!?；; 与换行），保留原文字符，不做任何截断。"""
    return [p.strip() for p in _SENTENCE_BOUNDARY.split(text) if p and p.strip()]


@dataclass
class Doc:
    doc_id: str
    title: str
    version_date: str
    topic: str
    sections: list[dict]

    @property
    def body_text(self) -> str:
        return "\n".join(f"{s['heading']}\n{s['text']}" for s in self.sections)

    def as_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "version_date": self.version_date,
            "topic": self.topic,
            "sections": self.sections,
        }


class Corpus:
    """从目录加载 D*.json 文档与可选的 search_config.json。"""

    def __init__(self, docs_dir: str | Path):
        self.docs_dir = Path(docs_dir)
        if not self.docs_dir.is_dir():
            raise FileNotFoundError(f"语料目录不存在: {self.docs_dir}")
        self.config = dict(DEFAULT_CONFIG)
        cfg = self.docs_dir / "search_config.json"
        if cfg.is_file():
            self.config.update(json.loads(cfg.read_text(encoding="utf-8-sig")))
        self.docs: dict[str, Doc] = {}
        for p in sorted(self.docs_dir.glob("D*.json")):
            data = json.loads(p.read_text(encoding="utf-8-sig"))
            doc = Doc(
                doc_id=data["doc_id"],
                title=data["title"],
                version_date=data["version_date"],
                topic=data.get("topic", ""),
                sections=data["sections"],
            )
            self.docs[doc.doc_id] = doc
        if not self.docs:
            raise ValueError(f"语料目录中没有 D*.json 文档: {self.docs_dir}")
        self._body_bigrams = {d: bigrams(doc.body_text) for d, doc in self.docs.items()}
        self._title_bigrams = {d: bigrams(doc.title) for d, doc in self.docs.items()}

    def get(self, doc_id: str) -> Doc | None:
        return self.docs.get(doc_id)

    def list_docs(self) -> list[dict]:
        return [
            {"doc_id": d.doc_id, "title": d.title, "version_date": d.version_date, "topic": d.topic}
            for d in self.docs.values()
        ]

    def corpus_hash(self) -> str:
        h = hashlib.sha256()
        for p in sorted(self.docs_dir.glob("*.json")):
            h.update(p.name.encode("utf-8"))
            h.update(p.read_bytes())
        return h.hexdigest()

    def search(self, query: str, top_k: int | None = None) -> dict:
        """字符 bigram 覆盖率评分检索。score = (body_cov + tw * title_cov) / (1 + tw)。"""
        cfg = self.config
        k = cfg["top_k_default"] if top_k is None else int(top_k)
        k = max(1, min(k, int(cfg["top_k_max"])))
        q = bigrams(query)
        if not q:
            return {"results": [], "best_score": 0.0, "empty_reason": "query too short"}
        tw = float(cfg["title_weight"])
        scored: list[tuple[float, Doc]] = []
        for doc_id, doc in self.docs.items():
            body_cov = len(q & self._body_bigrams[doc_id]) / len(q)
            title_cov = len(q & self._title_bigrams[doc_id]) / len(q)
            score = (body_cov + tw * title_cov) / (1 + tw)
            scored.append((score, doc))
        best_score = max(s for s, _ in scored)
        hits = [(s, d) for s, d in scored if s >= float(cfg["threshold"])]
        hits.sort(key=lambda x: (-x[0], x[1].doc_id))
        results = []
        for score, doc in hits[:k]:
            snippet, heading = self._best_snippet(doc, q)
            results.append(
                {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "version_date": doc.version_date,
                    "snippet": snippet,
                    "section": heading,
                    "score": round(score, 4),
                }
            )
        return {"results": results, "best_score": round(best_score, 4)}

    def _best_snippet(self, doc: Doc, q: set[str]) -> tuple[str, str]:
        """返回原文整句（或小节标题）作为命中片段，绝不截断、不加省略号。"""
        best: tuple[int, int, str, str] | None = None
        order = 0
        for sec in doc.sections:
            for sentence in split_sentences(sec["text"]):
                overlap = len(q & bigrams(sentence))
                key = (overlap, -len(sentence), order, sentence, sec["heading"])
                if best is None or (key[0], key[1], -key[2]) > (best[0], best[1], -best[2]):
                    best = (overlap, -len(sentence), order, sentence, sec["heading"])
                order += 1
        if best is not None and best[0] > 0:
            return best[3], best[4]
        # 正文无命中时退化到标题候选
        for sec in doc.sections:
            overlap = len(q & bigrams(sec["heading"]))
            if best is None or (overlap, -len(sec["heading"])) > (best[0], best[1]):
                best = (overlap, -len(sec["heading"]), 0, sec["heading"], sec["heading"])
        if best is None:
            sec = doc.sections[0]
            return sec["text"], sec["heading"]
        return best[3], best[4]
