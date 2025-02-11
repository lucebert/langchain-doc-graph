from typing import List

from typing_extensions import TypedDict


class InputState(TypedDict):
    """Represents the input.

    Attributes:
        question: str
    """

    question: str


class GraphState(InputState):
    """Represents the state of our graph.

    Attributes:
        generation: LLM generation
        documents: list of documents
    """

    generation: str
    total_iterations: int
    documents: List[str]
