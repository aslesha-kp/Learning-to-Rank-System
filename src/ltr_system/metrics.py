from __future__ import annotations

import math

import numpy as np

from .domain import Query, RankedList


class Evaluator:
    def __init__(self, k: int = 10):
        self.k = k

    def evaluate(self, truth: list[Query], predictions: list[RankedList]) -> dict[str, float]:
        truth_map = {q.qid: q for q in truth}
        pred_map = {r.qid: r for r in predictions}

        common_qids = sorted(set(truth_map).intersection(pred_map))
        if not common_qids:
            raise ValueError("No overlapping query IDs between truth and predictions.")

        ndcg_scores = []
        map_scores = []
        for qid in common_qids:
            ndcg_scores.append(ndcg_at_k(truth_map[qid], pred_map[qid], self.k))
            map_scores.append(average_precision(truth_map[qid], pred_map[qid]))

        return {
            f"ndcg@{self.k}": float(np.mean(ndcg_scores)),
            "map": float(np.mean(map_scores)),
        }


def _dcg(relevances: list[float], k: int) -> float:
    score = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        score += (2**rel - 1) / math.log2(i + 1)
    return score


def ndcg_at_k(truth: Query, pred: RankedList, k: int = 10) -> float:
    rel_by_doc = {d.doc_id: d.relevance for d in truth.documents}
    predicted_rels = [rel_by_doc.get(d.doc_id, 0.0) for d in pred.documents]
    ideal_rels = sorted((d.relevance for d in truth.documents), reverse=True)

    dcg = _dcg(predicted_rels, k)
    idcg = _dcg(ideal_rels, k)
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(truth: Query, pred: RankedList) -> float:
    rel_by_doc = {d.doc_id: d.relevance for d in truth.documents}
    binary_rels = [1 if rel_by_doc.get(d.doc_id, 0.0) > 0 else 0 for d in pred.documents]

    num_relevant = sum(1 for d in truth.documents if d.relevance > 0)
    if num_relevant == 0:
        return 0.0

    hit_count = 0
    precision_sum = 0.0
    for i, rel in enumerate(binary_rels, start=1):
        if rel:
            hit_count += 1
            precision_sum += hit_count / i

    return precision_sum / num_relevant
