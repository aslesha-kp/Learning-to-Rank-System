from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Document:
    doc_id: str
    qid: str
    relevance: float


@dataclass
class Query:
    qid: str
    documents: list[Document]

    def __post_init__(self) -> None:
        if not self.documents:
            raise ValueError(f"Query {self.qid} has no documents.")
        for doc in self.documents:
            if doc.qid != self.qid:
                raise ValueError("Document query ID mismatch.")


@dataclass
class RankedList:
    qid: str
    documents: list[Document]

    @classmethod
    def from_scored_documents(
        cls,
        qid: str,
        docs_and_scores: Iterable[tuple[Document, float]],
    ) -> "RankedList":
        ranked = sorted(docs_and_scores, key=lambda x: x[1], reverse=True)
        return cls(qid=qid, documents=[doc for doc, _ in ranked])
