### Nodes

from langgraph.graph import END, START, StateGraph

from self_rag.nodes.answer_grader import answer_grader
from self_rag.nodes.generate import rag_chain, format_docs
from self_rag.nodes.hallucination_grader import hallucination_grader
from self_rag.nodes.question_rewriter import question_rewriter
from self_rag.nodes.retrieval_grader import retrieval_grader
from self_rag.state import GraphState, InputState
from shared import retrieval
from shared.configuration import BaseConfiguration


def retrieve(state, *, config):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    with retrieval.make_retriever(config) as retriever:
        documents = retriever.invoke(question)
        return {"documents": documents, "question": question}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    total_iterations = state.get("total_iterations", 0)

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation, "total_iterations": total_iterations + 1}




workflow = StateGraph(GraphState, input=InputState, config_schema=BaseConfiguration)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("generate", generate)  # generatae

# Build graph
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile
graph = workflow.compile()
graph.name = "SimpleRag"
