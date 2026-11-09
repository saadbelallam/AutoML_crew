import subprocess
import sys
from pathlib import Path

from agents.data_summary import summarize_dataset
from agents.graph import build_graph
from harness.sandbox import run_in_sandbox

DATA_CSV = "data/titanic.csv"
TARGET = "Survived"
METRIC = "roc_auc"
BASELINE = "solutions/baseline.py"

if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
    sys.exit("Docker is not running. Start Docker Desktop and try again.")

baseline = run_in_sandbox(BASELINE, METRIC)
if baseline["status"] != "ok":
    sys.exit(f"Baseline evaluation failed:\n{baseline['error']}")
print(f"Baseline {METRIC}: {baseline['mean']:.4f} ± {baseline['std']:.4f}")

initial_state = {
    "data_summary": summarize_dataset(DATA_CSV, TARGET),
    "metric": METRIC,
    "best_code": Path(BASELINE).read_text(encoding="utf-8"),
    "best_score": baseline["mean"],
    "ideas": [],
    "results": [],
    "round": 0,
    "max_rounds": 3,
    "patience": 2,
    "stale_rounds": 0,
}

graph = build_graph()
final_state = graph.invoke(initial_state, config={"recursion_limit": 100})

print(f"\nFinished after {final_state['round']} rounds.")
print(f"Baseline {METRIC}: {baseline['mean']:.4f}")
print(f"Final best {METRIC}: {final_state['best_score']:.4f}")