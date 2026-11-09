import json
import subprocess
import uuid
from pathlib import Path

IMAGE = "automl-sandbox"


def failed_result(solution_path, metric, message):
    return {
        "solution": str(solution_path),
        "metric": metric,
        "status": "error",
        "error": message,
        "fold_scores": [],
        "mean": None,
        "std": None,
    }


def run_in_sandbox(solution_path, metric, project_dir=".", timeout_s=300, memory="2g", cpus=2):
    project_dir = Path(project_dir).resolve()
    solution_name = Path(solution_path).name
    container_name = f"eval-{uuid.uuid4().hex[:8]}"

    cmd = [
        "docker", "run", "--rm", "--name", container_name,
        "--network", "none",
        "--memory", memory, "--cpus", str(cpus), "--pids-limit", "256",
        "--read-only", "--tmpfs", "/tmp",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "-v", f"{project_dir / 'data'}:/work/data:ro",
        "-v", f"{project_dir / 'solutions'}:/work/solutions:ro",
        IMAGE,
        "python", "-m", "harness.evaluate", f"solutions/{solution_name}", "--metric", metric,
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", container_name], capture_output=True)
        return failed_result(solution_path, metric, f"timed out after {timeout_s} seconds")

    if proc.returncode != 0:
        if proc.returncode == 137:
            message = f"killed by the sandbox, most likely out of memory (limit {memory})"
        else:
            message = proc.stderr[-2000:] or f"container exited with code {proc.returncode}"
        return failed_result(solution_path, metric, message)

    return json.loads(proc.stdout)