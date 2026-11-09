from langgraph.graph import StateGraph, START, END

from agents.nodes import worker_node, dispatch_workers, report_node
from agents.state import LabState

def build_graph():
    builder = StateGraph(LabState)

    builder.add_node("worker", worker_node)
    builder.add_node("report", report_node)

    builder.add_conditional_edges(START, dispatch_workers, ["worker"])
    builder.add_edge("worker", "report")
    builder.add_edge("report", END)

    return builder.compile()