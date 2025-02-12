### Nodes

from langchain import hub
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from shared import retrieval
from shared.configuration import BaseConfiguration
from simple_rag.state import GraphState, InputState


def retrieve(state: GraphState, *, config) -> dict[str, list[str] | str]: 
    """Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    # Extract human messages and concatenate them
    question = " ".join(msg.content for msg in state.messages if isinstance(msg, HumanMessage))

    # Retrieval
    with retrieval.make_retriever(config) as retriever:
        documents = retriever.invoke(question)
        return {"documents": documents, "message": state.messages}


async def generate(state: GraphState):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    messages = state.messages
    documents = state.documents

    # RAG generation
    # Prompt
    prompt = hub.pull("self-rag")

    # LLM
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    

    # Chain
    rag_chain = prompt + messages | llm
    response = await rag_chain.ainvoke({"context" : documents})
    return {"messages": [response], "documents": documents}


workflow = StateGraph(GraphState, input=InputState, config_schema=BaseConfiguration)

# Define the nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

# Build graph
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile
graph = workflow.compile()
graph.name = "SimpleRag"
