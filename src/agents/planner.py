from pydantic import BaseModel, Field

from agents.llm import get_llm
from agents.state import LabState
from harness.evaluate import METRICS

N_IDEAS = 3

class Plan(BaseModel):
    ideas: list[str] = Field(
        description="Experiment ideas, each one a single concrete change to the current best solution.",
    )

PLANNER_PROMPT = """You are leading a team of machine learning engineers on a tabular {task} problem.
The metric is {metric} ({direction} is better).

Propose {n_ideas} experiment ideas for the next round. Each idea will be given to a
different engineer, who will apply it as ONE change to the current best solution.

Good ideas:
- are concrete and specific (name the columns, the transformation, the model)
- change one thing only, so we can tell what helped
- build on what worked, and do not repeat what was already tried
- are different from each other, so the round explores several directions

Dataset description:
{data_summary}

Current best solution ({metric} = {best_score:.4f}):
```python
{best_code}
```

Experiments tried so far:
{history}"""

def format_history(results):
    if not results:
        return "None yet."
    lines = []
    for r in results:
        score = f"{r['score']:.4f}" if r["status"] == "ok" else "failed"
        lines.append(f"- round {r['round']} | {score} | {r['idea']}")
    return "\n".join(lines)


def planner_node(state: LabState) -> dict:
    prompt = PLANNER_PROMPT.format(
        task="classification",
        metric=state["metric"],
        direction="higher" if METRICS[state["metric"]][1] else "lower",
        n_ideas=N_IDEAS,
        data_summary=state["data_summary"],
        best_score=state["best_score"],
        best_code=state["best_code"],
        history=format_history(state["results"]),
    )

    planner = get_llm().with_structured_output(Plan)
    plan = planner.invoke(prompt)

    return {"ideas": plan.ideas[:N_IDEAS], "round": state["round"] + 1}