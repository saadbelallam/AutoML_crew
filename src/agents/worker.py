from langchain.agents import create_agent

from agents.llm import get_llm
from agents.tools import make_evaluate_tool
from langgraph.errors import GraphRecursionError

from harness.evaluate import METRICS

SYSTEM_PROMPT = """You are a machine learning engineer working on a tabular {task} problem.

Your job: you are given the current best solution and ONE experiment idea.
Apply the idea as a change to that solution, then score the result with the
evaluate_solution tool.

Rules for the solution file:
- It must define fit_predict(X_train, y_train, X_valid).
- Return a 1D array with the probability of the positive class for each row of X_valid.
  Use predict_proba(X_valid)[:, 1], never predict(X_valid).
- Learn all preprocessing on X_train only, ideally inside a scikit-learn Pipeline.
- Handle missing values. X_valid can contain categories never seen in X_train.
- Only use numpy, pandas and scikit-learn. No network access, no reading or writing files.
- The target column is not in X_train or X_valid.
- Implement exactly the idea you are given and nothing else. Do not add other
  features or change the model unless the idea says so.

If the tool returns an error, read it, fix the code and call the tool again.
Once you have a working score, stop and reply with a two-sentence summary of what you implemented.

Dataset description:
{data_summary}"""

def run_worker(experiment_id, idea, base_code, data_summary, metric, task="classification", max_attempts=4):

    evaluate_solution, attempts = make_evaluate_tool(experiment_id, metric,max_attempts)

    agent = create_agent(
        model= get_llm(),
        system_prompt=SYSTEM_PROMPT.format(task=task, data_summary=data_summary),
        tools=[evaluate_solution],
    )

    summary = None
    user_message = (
        f"Current best solution:\n```python\n{base_code}\n```\n\n"
        f"Experiment idea: {idea}"
    )

    messages = []
    try:
        for state in agent.stream(
            {"messages": [{"role": "user", "content": user_message}]},
            config={"recursion_limit": 2 * max_attempts + 4},
            stream_mode="values",
        ):
            messages = state["messages"]
        summary = messages[-1].text
    except GraphRecursionError:
        summary = "Stopped: the agent kept going after its attempts ran out."

    successful_attempts = [a for a in attempts if a["status"] == "ok"]
    best = None
    if successful_attempts:
        higher_is_better = METRICS[metric][1]
        pick = max if higher_is_better else min
        best = pick(successful_attempts, key=lambda a: a["mean"])

    return {
        "experiment_id": experiment_id,
        "idea": idea,
        "summary": summary,
        "n_attempts": len(attempts),
        "best": best,
        "attempts": attempts,
        "messages": messages,
    }

