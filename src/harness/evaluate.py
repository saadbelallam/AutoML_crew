import json
import runpy
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error

from harness.folds import load_folds


def load_solution(path):
    namespace = runpy.run_path(path)

    if "fit_predict" not in namespace:
        raise AttributeError(f"{path} must define fit_predict(X_train, y_train, X_valid)")
    return namespace["fit_predict"]


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


# name -> (function, higher_is_better)
METRICS = {
    "roc_auc": (roc_auc_score, True),
    "log_loss": (log_loss, False),
    "rmse": (rmse, False),
}


def check_predictions(preds, n_expected, task):
    preds = np.asarray(preds, dtype=float)

    if preds.ndim != 1:
        raise ValueError(f"predictions must be 1D, got shape {preds.shape}")
    if len(preds) != n_expected:
        raise ValueError(f"expected {n_expected} predictions, got {len(preds)}")
    if not np.all(np.isfinite(preds)):
        raise ValueError("predictions contain NaN or infinite values")

    if task == "classification":
        if preds.min() < 0 or preds.max() > 1:
            raise ValueError("classification predictions must be probabilities between 0 and 1")
        if set(np.unique(preds)) <= {0.0, 1.0}:
            raise ValueError("predictions are hard 0/1 labels, return probabilities "
                             "(predict_proba(X_valid)[:, 1]) instead of predict(X_valid)")

    return preds


def evaluate(solution_path, folds_path, metric):
    spec = load_folds(folds_path)
    df = pd.read_csv(spec["train_csv"])
    X = df.drop(columns=[spec["target"]])
    y = df[spec["target"]]
    metric_fn, higher_is_better = METRICS[metric]

    result = {
        "solution": str(solution_path),
        "metric": metric,
        "higher_is_better": higher_is_better,
        "status": "ok",
        "error": None,
        "fold_scores": [],
        "mean": None,
        "std": None,
    }

    start = time.time()
    try:
        fit_predict = load_solution(solution_path)
        for fold in spec["folds"]:
            train_rows, valid_rows = fold["train"], fold["valid"]
            preds = fit_predict(X.iloc[train_rows].copy(), y.iloc[train_rows].copy(),
                                X.iloc[valid_rows].copy())
            preds = check_predictions(preds, len(valid_rows), spec["task"])
            result["fold_scores"].append(float(metric_fn(y.iloc[valid_rows], preds)))
    except Exception:
        result["status"] = "error"
        result["error"] = traceback.format_exc()[-2000:]

    result["runtime_s"] = round(time.time() - start, 2)
    if result["status"] == "ok":
        result["mean"] = float(np.mean(result["fold_scores"]))
        result["std"] = float(np.std(result["fold_scores"]))
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score a solution file on the fixed folds")
    parser.add_argument("solution", help="path to the solution .py file")
    parser.add_argument("--folds", default="data/folds.json")
    parser.add_argument("--metric", required=True, choices=list(METRICS))
    args = parser.parse_args()

    result = evaluate(args.solution, args.folds, args.metric)
    print(json.dumps(result, indent=2))