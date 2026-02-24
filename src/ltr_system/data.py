from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .domain import Document, Query


@dataclass
class LTRDataset:
    frame: pd.DataFrame
    feature_cols: list[str]

    def X(self) -> np.ndarray:
        return self.frame[self.feature_cols].to_numpy(dtype=np.float32)

    def y(self) -> np.ndarray:
        return self.frame["label"].to_numpy(dtype=np.float32)

    def qids(self) -> np.ndarray:
        return self.frame["qid"].to_numpy()

    def groups(self) -> list[int]:
        counts = self.frame.groupby("qid", sort=False).size()
        return counts.astype(int).tolist()

    def queries(self) -> list[Query]:
        result: list[Query] = []
        for qid, qdf in self.frame.groupby("qid", sort=False):
            docs = [
                Document(
                    doc_id=str(row.doc_id),
                    qid=str(qid),
                    relevance=float(row.label),
                )
                for row in qdf.itertuples(index=False)
            ]
            result.append(Query(qid=str(qid), documents=docs))
        return result


class LETORParser:
    def __init__(self, num_features: int = 136):
        self.num_features = num_features

    def parse_file(self, file_path: str | Path) -> LTRDataset:
        rows: list[dict[str, object]] = []
        file_path = Path(file_path)

        with file_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                parsed = self._parse_line(line, i)
                if parsed is not None:
                    rows.append(parsed)

        if not rows:
            raise ValueError(f"No valid rows parsed from {file_path}.")

        frame = pd.DataFrame(rows)
        feature_cols = [f"f{j}" for j in range(1, self.num_features + 1)]

        for col in feature_cols:
            if col not in frame.columns:
                frame[col] = 0.0

        frame = frame[["label", "qid", "doc_id", *feature_cols]]
        frame = frame.sort_values(["qid", "doc_id"]).reset_index(drop=True)

        return LTRDataset(frame=frame, feature_cols=feature_cols)

    def _parse_line(self, line: str, idx: int) -> dict[str, object] | None:
        clean = line.strip()
        if not clean:
            return None

        content, comment = (clean.split("#", 1) + [""])[:2]
        tokens = content.strip().split()
        if len(tokens) < 2:
            raise ValueError(f"Malformed LETOR line at index {idx}: {line}")

        label = float(tokens[0])
        qid_token = tokens[1]
        if not qid_token.startswith("qid:"):
            raise ValueError(f"Missing qid token at line {idx}: {line}")
        qid = qid_token.split(":", 1)[1]

        row: dict[str, object] = {
            "label": label,
            "qid": qid,
            "doc_id": self._extract_doc_id(comment, idx),
        }

        for token in tokens[2:]:
            if ":" not in token:
                continue
            fidx_str, fval_str = token.split(":", 1)
            try:
                fidx = int(fidx_str)
            except ValueError:
                continue
            if 1 <= fidx <= self.num_features:
                row[f"f{fidx}"] = float(fval_str)

        return row

    @staticmethod
    def _extract_doc_id(comment: str, idx: int) -> str:
        parts = comment.strip().split()
        for part in parts:
            if part.startswith("docid="):
                return part.split("=", 1)[1]
        return f"auto_doc_{idx}"


def load_fold(fold_dir: str | Path, parser: LETORParser) -> tuple[LTRDataset, LTRDataset, LTRDataset]:
    fold_dir = Path(fold_dir)
    train = parser.parse_file(fold_dir / "train.txt")
    valid = parser.parse_file(fold_dir / "vali.txt")
    test = parser.parse_file(fold_dir / "test.txt")
    return train, valid, test
