from langgraph.graph import StateGraph, START, END

from agents.nodes import should_continue, worker_node, dispatch_workers, report_node, evaluator_node
from agents.planner import planner_node
from agents.state import LabState

def build_graph():
    builder = StateGraph(LabState)

    builder.add_node("planner", planner_node)
    builder.add_node("worker", worker_node)
    builder.add_node("report", report_node)
    builder.add_node("evaluator", evaluator_node)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", dispatch_workers, ["worker"])
    builder.add_edge("worker", "evaluator")
    builder.add_edge("evaluator", "report")
    builder.add_conditional_edges("report", should_continue, {"planner": "planner", "end": END})

    return builder.compile()