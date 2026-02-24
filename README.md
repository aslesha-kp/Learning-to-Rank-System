# Learning-to-Rank System

End-to-end Learning-to-Rank (LTR) project for LETOR-style datasets such as **MSLR-WEB10K**.

Implements:
- Pointwise model: `LinearRegression` (scikit-learn)
- Listwise model: `LambdaMART` (`LGBMRanker` from LightGBM)
- Baseline model: feature-aggregation score

Includes a modular OOP evaluation framework with:
- `Query`, `Document`, `RankedList`, `Evaluator`
- Metrics: `NDCG@10`, `MAP`

## Project Structure

- `src/ltr_system/domain.py`: ranking abstractions
- `src/ltr_system/metrics.py`: NDCG/MAP computation
- `src/ltr_system/data.py`: LETOR parser and fold loading
- `src/ltr_system/models.py`: baseline, pointwise, listwise models
- `src/ltr_system/experiment.py`: reproducible experiment runner
- `scripts/run_experiment.py`: CLI entrypoint

## Data Format

Expects LETOR/MSLR lines like:

```text
2 qid:101 1:0.32 2:0.18 ... 136:0.91 #docid=ABC
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Where To Put MSLR-WEB10K

Recommended layout:

```text
Project 1/
  data/
    MSLR-WEB10K/
      Fold1/
        train.txt
        vali.txt
        test.txt
      Fold2/
      Fold3/
      Fold4/
      Fold5/
```

You can keep the dataset elsewhere too; just pass absolute/relative paths in commands.

Run a single split (explicit files):

```bash
python scripts/run_experiment.py \
  --train data/MSLR-WEB10K/Fold1/train.txt \
  --valid data/MSLR-WEB10K/Fold1/vali.txt \
  --test data/MSLR-WEB10K/Fold1/test.txt \
  --run-name fold1_single \
  --seed 42
```

Run one fold directory containing `train.txt`, `vali.txt`, `test.txt`:

```bash
python scripts/run_experiment.py \
  --fold-dir data/MSLR-WEB10K/Fold1 \
  --run-name fold1 \
  --seed 42
```

Batch all 5 folds:

```bash
python scripts/run_experiment.py \
  --fold-dir data/MSLR-WEB10K/Fold1 \
  --fold-dir data/MSLR-WEB10K/Fold2 \
  --fold-dir data/MSLR-WEB10K/Fold3 \
  --fold-dir data/MSLR-WEB10K/Fold4 \
  --fold-dir data/MSLR-WEB10K/Fold5 \
  --run-name all_folds \
  --seed 42
```

## Result Files

Each run writes to `results/<run-name>/` (or a timestamped folder if `--run-name` is omitted):

- `metadata.json`: config and paths used
- Single split mode: `single_split_results.csv`
- Fold mode: `per_fold_results.csv`, `summary_results.csv`

## Reproducibility

- Fixed `random_state` for model training
- Deterministic data sorting by `(qid, doc_id)`
- Fold-level and overall aggregate reporting

## Notes

- For MSLR-WEB10K, keep `--num-features 136` (default).
- `LGBMRanker` requires group sizes per query for train/valid.
