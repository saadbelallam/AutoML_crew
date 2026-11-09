from agents.state import LabState, WorkerInput
from agents.worker import run_worker

from langgraph.types import Send


def worker_node(input: WorkerInput) -> dict:
    out = run_worker(
        experiment_id=input["experiment_id"],
        idea=input["idea"],
        base_code=input["base_code"],
        data_summary=input["data_summary"],
        metric=input["metric"],
    )

    result = {
        "round": input["round"],
        "experiment_id": input["experiment_id"],
        "idea": input["idea"],
        "summary": out["summary"],
        "n_attempts": out["n_attempts"],
        "status": "ok" if out["best"] else "failed",
        "score": out["best"]["mean"] if out["best"] else None,
        "std": out["best"]["std"] if out["best"] else None,
        "path": out["best"]["path"] if out["best"] else None,
    }

    return {"results": [result] }

def dispatch_workers(state: LabState) -> list[Send]:
    sends = []
    for i, idea in enumerate(state["ideas"], start=1):
        worker_input: WorkerInput = {
            "experiment_id": f"r{state['round']}_e{i}",
            "idea": idea,
            "base_code": state["best_code"],
            "data_summary": state["data_summary"],
            "metric": state["metric"],
            "round": state["round"],
        }
        ## Send("worker", data) is an instruction to LangGraph: 
        # “run the node named worker with this input
        sends.append(Send("worker", worker_input))
    return sends

def report_node(state: LabState) -> dict:
    this_round = [r for r in state["results"] if r["round"] == state["round"]]
    this_round.sort(key=lambda r: r["score"] if r["score"] is not None else float("-inf"),
                    reverse=True)

    print(f"\nRound {state['round']} (best so far: {state['best_score']:.4f})")
    for r in this_round:
        score = f"{r['score']:.4f} ± {r['std']:.4f}" if r["score"] is not None else "failed"
        print(f"  {r['experiment_id']}  {score}  ({r['n_attempts']} attempts)  {r['idea']}")
    return {}