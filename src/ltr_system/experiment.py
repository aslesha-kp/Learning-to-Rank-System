from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import LETORParser, LTRDataset, load_fold
from .domain import Document, RankedList
from .metrics import Evaluator
from .models import BaselineRanker, LambdaMARTRanker, PointwiseLinearRanker, Ranker


@dataclass
class ExperimentConfig:
    num_features: int = 136
    k: int = 10
    random_state: int = 42


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.parser = LETORParser(num_features=config.num_features)
        self.evaluator = Evaluator(k=config.k)

    def run_single_split(
        self,
        train_path: str | Path,
        valid_path: str | Path,
        test_path: str | Path,
    ) -> pd.DataFrame:
        train = self.parser.parse_file(train_path)
        valid = self.parser.parse_file(valid_path)
        test = self.parser.parse_file(test_path)
        return self._run_models(train, valid, test, split_name="single")

    def run_folds(self, fold_dirs: list[str | Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
        fold_results: list[pd.DataFrame] = []
        for fold_dir in fold_dirs:
            train, valid, test = load_fold(fold_dir, self.parser)
            fold_name = Path(fold_dir).name
            fold_results.append(self._run_models(train, valid, test, split_name=fold_name))

        all_folds = pd.concat(fold_results, ignore_index=True)
        summary = (
            all_folds.groupby("model", as_index=False)[[f"ndcg@{self.config.k}", "map"]]
            .mean()
            .sort_values(f"ndcg@{self.config.k}", ascending=False)
            .reset_index(drop=True)
        )
        return all_folds, summary

    def _run_models(
        self,
        train: LTRDataset,
        valid: LTRDataset,
        test: LTRDataset,
        split_name: str,
    ) -> pd.DataFrame:
        model_factories = {
            "baseline": lambda: BaselineRanker(),
            "pointwise_linear": lambda: PointwiseLinearRanker(),
            "lambdamart": lambda: LambdaMARTRanker(random_state=self.config.random_state),
        }

        rows: list[dict[str, object]] = []
        truth_queries = test.queries()

        for model_name, factory in model_factories.items():
            try:
                model = factory()
                model.fit(train, valid)
                scores = model.predict(test)
                ranked_lists = _to_ranked_lists(test, scores)
                metrics = self.evaluator.evaluate(truth_queries, ranked_lists)
                rows.append({"split": split_name, "model": model_name, **metrics, "error": ""})
            except Exception as exc:
                rows.append(
                    {
                        "split": split_name,
                        "model": model_name,
                        f"ndcg@{self.config.k}": np.nan,
                        "map": np.nan,
                        "error": str(exc),
                    }
                )

        return pd.DataFrame(rows)


def _to_ranked_lists(dataset: LTRDataset, scores: np.ndarray) -> list[RankedList]:
    frame = dataset.frame.copy()
    frame["score"] = scores

    ranked: list[RankedList] = []
    for qid, qdf in frame.groupby("qid", sort=False):
        docs_scores = [
            (
                Document(doc_id=str(row.doc_id), qid=str(qid), relevance=float(row.label)),
                float(row.score),
            )
            for row in qdf.itertuples(index=False)
        ]
        ranked.append(RankedList.from_scored_documents(qid=str(qid), docs_and_scores=docs_scores))

    return ranked
