from pathlib import Path

from langchain.tools import tool
from harness.sandbox import run_in_sandbox

SOLUTIONS_DIR = Path("solutions")

def make_evaluate_tool(experiment_id, metric, max_attempts=4):
    attempts = []

    @tool
    def evaluate_solution(code: str) -> str:
        """Save a solution file and score it on the fixed cross-validation folds.

        `code` must be a complete Python file that defines
        fit_predict(X_train, y_train, X_valid) and returns one probability
        per row of X_valid. Returns the score, or the error to fix.
        """
        if len(attempts) >= max_attempts:
            return "No attempts left. Stop and summarize what you tried."

        n = len(attempts) + 1
        path = SOLUTIONS_DIR / f"{experiment_id}_try{n}.py"
        path.write_text(code, encoding="utf-8")

        result = run_in_sandbox(path, metric)
        attempts.append({"attempt": n, "path": str(path), **result})

        if result["status"] == "ok":
            return f"Attempt {n}: {metric} = {result['mean']:.4f} ± {result['std']:.4f}"
        return f"Attempt {n} failed:\n{result['error']}\nFix the code and try again."

    return evaluate_solution, attempts