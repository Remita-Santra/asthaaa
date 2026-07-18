from langgraph.graph import StateGraph, START, END
from state import ASHAAgentState
import nodes

def build_graph():
    graph = StateGraph(ASHAAgentState)

    graph.add_node("ingest", nodes.ingest_node)
    graph.add_node("language_and_translate", nodes.language_and_translate_node)
    graph.add_node("extract_vitals", nodes.extract_vitals_node)
    graph.add_node("muac_analysis", nodes.muac_analysis_node)
    graph.add_node("maternal_risk", nodes.maternal_risk_node)
    graph.add_node("triage", nodes.triage_node)
    graph.add_node("guidance_generation", nodes.guidance_generation_node)

    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "language_and_translate")
    graph.add_edge("language_and_translate", "extract_vitals")

    graph.add_conditional_edges(
        "extract_vitals",
        nodes.route_by_patient_type,
        {
            "muac_analysis": "muac_analysis",
            "maternal_risk": "maternal_risk",
            "triage": "triage",
        },
    )

    graph.add_edge("muac_analysis", "triage")
    graph.add_edge("maternal_risk", "triage")
    graph.add_edge("triage", "guidance_generation")
    graph.add_edge("guidance_generation", END)

    return graph.compile()

asha_agent_graph = build_graph()