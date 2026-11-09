import pandas as pd


def summarize_dataset(csv_path, target, n_examples=3):
    df = pd.read_csv(csv_path)
    lines = [
        f"Dataset: {len(df)} rows, {df.shape[1] - 1} feature columns.",
        f"Target column: '{target}' (not included in X_train / X_valid).",
        f"Target distribution: {df[target].value_counts(normalize=True).round(3).to_dict()}",
        "",
        "Columns (name | dtype | missing | unique values | examples):",
    ]
    for col in df.columns:
        if col == target:
            continue
        s = df[col]
        examples = s.dropna().unique()[:n_examples].tolist()
        lines.append(
            f"- {col} | {s.dtype} | {s.isna().mean():.0%} | {s.nunique()} | {examples}"
        )
    return "\n".join(lines)