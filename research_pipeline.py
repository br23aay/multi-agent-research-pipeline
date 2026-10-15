import operator
import time  # Imported to add sleep delays for rate limiting
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from langchain_groq import ChatGroq


# --- State Schema ---

class ResearchState(TypedDict):
    query: str
    classification: str
    research_findings: str
    report: str
    human_approved: bool
    iteration_count: int
    quality_score: int
    quality_feedback: str    


# --- LLM Setup ---

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)


# --- Agent Nodes ---

def router_node(state: ResearchState) -> dict:
    """Classify the research query into a category."""
    print("[System] Waiting for rate limits...")
    time.sleep(3)  # Brief pause before initial run
    response = llm.invoke(
        f"Classify this research query into exactly one category: "
        f"'technical', 'business', or 'general'. "
        f"Respond with only the category word.\n\n"
        f"Query: {state['query']}"
    )
    classification = response.content.strip().lower()
    print(f"[Router] Classified as: {classification}")
    return {"classification": classification}


def researcher_node(state: ResearchState) -> dict:
    """Conduct research based on the classification."""
    system_prompts = {
        "technical": "You are a senior technical researcher. Provide detailed technical analysis with specific technologies, architectures, and implementation details.",
        "business": "You are a business strategy researcher. Focus on market trends, ROI, competitive advantages, and business impact.",
        "general": "You are a research analyst. Provide a balanced overview covering key facts, context, and practical implications.",
    }
    system_prompt = system_prompts.get(state["classification"], system_prompts["general"])

    # If there's quality feedback, incorporate it
    user_content = f"Research this topic thoroughly:\n\n{state['query']}"
    if state.get("quality_feedback"):
        user_content += (
            f"\n\nPrevious attempt received this feedback: {state['quality_feedback']}"
            f"\nPlease address the feedback and provide more comprehensive research."
        )

    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
    )
    iteration = state.get("iteration_count", 0) + 1
    print(f"[Researcher] Completed research iteration {iteration} ({len(response.content)} chars)")
    return {"research_findings": response.content, "iteration_count": iteration}

def synthesizer_node(state: ResearchState) -> dict:
    """Synthesize research findings into a structured report."""
    print("[System] Waiting for rate limits...")
    time.sleep(6)  # Pause to let rate limit window recover after large research output
    response = llm.invoke(
        f"Based on the following research findings, create a structured report "
        f"with these sections:\n"
        f"1. Summary (2-3 sentences)\n"
        f"2. Key Points (bullet list)\n"
        f"3. Recommendations (actionable next steps)\n\n"
        f"Research findings:\n{state['research_findings']}"
    )
    print(f"[Synthesizer] Report generated")
    return {"report": response.content}

def quality_check_node(state: ResearchState) -> dict:
    """Evaluate report quality and provide feedback."""
    response = llm.invoke(
        f"Evaluate this research report on two criteria:\n"
        f"1. Completeness (does it cover the topic thoroughly?)\n"
        f"2. Actionability (are the recommendations specific and useful?)\n\n"
        f"Score from 1-10 and provide brief feedback.\n"
        f"Format your response as:\n"
        f"SCORE: <number>\n"
        f"FEEDBACK: <one sentence>\n\n"
        f"Report:\n{state['report']}"
    )

    content = response.content
    score = 7
    feedback = ""
    for line in content.split("\n"):
        if line.strip().startswith("SCORE:"):
            try:
                score = int(line.split(":")[1].strip())
            except ValueError:
                score = 7
        elif line.strip().startswith("FEEDBACK:"):
            feedback = line.split(":", 1)[1].strip()

    print(f"[Quality Check] Score: {score}/10 | Iteration: {state.get('iteration_count', 1)}")
    return {"quality_score": score, "quality_feedback": feedback}

def human_review_node(state: ResearchState) -> dict:
    """Pause for human approval of the report."""
    warning = ""
    if state.get("iteration_count", 0) >= 3:
        warning = " (WARNING: Max iterations reached, quality may be limited)"

    decision = interrupt(
        {
            "report": state["report"],
            "quality_score": state.get("quality_score", "N/A"),
            "message": f"Please review the report above{warning}. Type 'approve' to accept or 'revise' to request changes.",
        }
    )
    if decision == "approve":
        print("[Human Review] Report approved!")
        return {"human_approved": True}
    else:
        print("[Human Review] Report rejected.")
        return {"human_approved": False}

# --- Routing Logic ---

def route_after_research(state: ResearchState) -> Literal["synthesizer_node", END]:
    """Route based on whether research findings are sufficient."""
    if state.get("research_findings") and len(state["research_findings"]) > 50:
        return "synthesizer_node"
    return END

def route_after_quality(state: ResearchState) -> Literal["human_review_node", "researcher_node"]:
    """Route based on quality score and iteration count."""
    score = state.get("quality_score", 7)
    iterations = state.get("iteration_count", 1)

    if score >= 7:
        return "human_review_node"
    elif iterations >= 3:
        # Max retries reached, proceed with warning
        return "human_review_node"
    else:
        # Send back for more research
        return "researcher_node"

# --- Build the Graph ---

builder = StateGraph(ResearchState)

# Add nodes
builder.add_node("router_node", router_node)
builder.add_node("researcher_node", researcher_node)
builder.add_node("synthesizer_node", synthesizer_node)
builder.add_node("quality_check_node", quality_check_node)
builder.add_node("human_review_node", human_review_node)

# Add edges
builder.add_edge(START, "router_node")
builder.add_edge("router_node", "researcher_node")
builder.add_conditional_edges("researcher_node", route_after_research, ["synthesizer_node", END])
builder.add_edge("synthesizer_node", "quality_check_node")
builder.add_conditional_edges("quality_check_node", route_after_quality, ["human_review_node", "researcher_node"])
builder.add_edge("human_review_node", END)

# Compile with checkpointer for human-in-the-loop
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# --- Main Execution ---

def run_pipeline():
    """Run the research pipeline with human-in-the-loop."""
    query = input("Enter your research question: ")

    config = {"configurable": {"thread_id": "research-session-1"}}

    # First invocation - runs until interrupt
    print("\n--- Starting Research Pipeline ---\n")
    result = graph.invoke({"query": query}, config=config)

    # Check if we hit an interrupt
    state = graph.get_state(config)
    while state.next:
        # We're at an interrupt - get the interrupt payload
        interrupt_value = state.tasks[0].interrupts[0].value
        print(f"\n--- HUMAN REVIEW REQUIRED ---")
        print(f"\nQuality Score: {interrupt_value.get('quality_score', 'N/A')}/10")
        print(f"\nReport:\n{interrupt_value['report']}")
        print(f"\n{interrupt_value['message']}")

        decision = input("\nYour decision: ").strip().lower()

        # Resume the graph with the human's decision
        result = graph.invoke(Command(resume=decision), config=config)
        state = graph.get_state(config)

    print("\n--- Pipeline Complete ---")
    print(f"\nFinal Report:\n{result.get('report', 'No report generated')}")
    print(f"Approved: {result.get('human_approved', False)}")
    print(f"Quality Score: {result.get('quality_score', 'N/A')}/10")
    print(f"Research Iterations: {result.get('iteration_count', 1)}")

if __name__ == "__main__":
    run_pipeline()
    