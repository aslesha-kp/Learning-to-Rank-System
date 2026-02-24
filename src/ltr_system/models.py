from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.linear_model import LinearRegression

from .data import LTRDataset


class Ranker(ABC):
    @abstractmethod
    def fit(self, train: LTRDataset, valid: LTRDataset | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, dataset: LTRDataset) -> np.ndarray:
        raise NotImplementedError


class BaselineRanker(Ranker):
    """Simple baseline: mean feature value per document."""

    def fit(self, train: LTRDataset, valid: LTRDataset | None = None) -> None:
        return

    def predict(self, dataset: LTRDataset) -> np.ndarray:
        return dataset.X().mean(axis=1)


class PointwiseLinearRanker(Ranker):
    def __init__(self) -> None:
        self.model = LinearRegression()

    def fit(self, train: LTRDataset, valid: LTRDataset | None = None) -> None:
        self.model.fit(train.X(), train.y())

    def predict(self, dataset: LTRDataset) -> np.ndarray:
        return self.model.predict(dataset.X())


class LambdaMARTRanker(Ranker):
    def __init__(
        self,
        random_state: int = 42,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
    ) -> None:
        try:
            from lightgbm import LGBMRanker
        except Exception as exc:
            raise RuntimeError(
                "LightGBM is unavailable. On macOS install OpenMP first: "
                "`brew install libomp`, then reinstall lightgbm in your venv."
            ) from exc

        self.model = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            random_state=random_state,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
        )

    def fit(self, train: LTRDataset, valid: LTRDataset | None = None) -> None:
        if valid is not None:
            self.model.fit(
                train.X(),
                train.y(),
                group=train.groups(),
                eval_set=[(valid.X(), valid.y())],
                eval_group=[valid.groups()],
                eval_at=[10],
            )
        else:
            self.model.fit(train.X(), train.y(), group=train.groups())

    def predict(self, dataset: LTRDataset) -> np.ndarray:
        return self.model.predict(dataset.X())
