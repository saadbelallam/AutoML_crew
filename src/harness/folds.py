import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

import hashlib
import json
from pathlib import Path


def make_folds(df, target, task, n_splits=5, seed=42):
    if task == "classification":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = []
    for train_idx, valid_idx in splitter.split(df, df[target]):
        folds.append({"train": train_idx.tolist(), "valid": valid_idx.tolist()})
    return folds



def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_folds(folds, out_path, train_csv, target, task, seed):
    spec = {
        "train_csv": str(train_csv),
        "data_sha256": file_sha256(train_csv),
        "target": target,
        "task": task,
        "seed": seed,
        "n_rows": len(folds[0]["train"]) + len(folds[0]["valid"]),
        "folds": folds,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(spec))
    return spec


def load_folds(folds_path):
    spec = json.loads(Path(folds_path).read_text())

    current_hash = file_sha256(spec["train_csv"])
    if current_hash != spec["data_sha256"]:
        raise ValueError(
            f"{spec['train_csv']} has changed since the folds were created. "
            f"Regenerate the folds before evaluating anything."
        )
    return spec

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create fixed CV folds for a dataset")
    parser.add_argument("--train", required=True, help="path to the training CSV")
    parser.add_argument("--target", required=True, help="name of the target column")
    parser.add_argument("--task", required=True, choices=["classification", "regression"])
    parser.add_argument("--out", default="data/folds.json")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.train)
    folds = make_folds(df, args.target, args.task, args.n_splits, args.seed)
    spec = save_folds(folds, args.out, args.train, args.target, args.task, args.seed)
    print(f"Saved {len(folds)} folds for {spec['n_rows']} rows to {args.out}")