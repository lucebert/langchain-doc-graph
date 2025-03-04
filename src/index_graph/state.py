"""State management for the index graph."""

from dataclasses import dataclass, field


@dataclass(kw_only=True)
class InputState:
    """The input state for the index graph."""
    
    url_site_map: str
    """The URL to the site map to index."""


# The index state defines the simple IO for the single-node index graph
@dataclass(kw_only=True)
class IndexState(InputState):
    """Represents the state for document indexing and retrieval.

    This class defines the structure of the index state, which includes
    the documents to be indexed and the retriever used for searching
    these documents.
    """

    urls_to_index: list[str] = field(default_factory=list)
    """The URLs to index."""

