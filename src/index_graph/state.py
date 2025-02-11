"""State management for the index graph."""

from dataclasses import dataclass


# The index state defines the simple IO for the single-node index graph
@dataclass(kw_only=True)
class IndexState:
    """Represents the state for document indexing and retrieval.

    This class defines the structure of the index state, which includes
    the documents to be indexed and the retriever used for searching
    these documents.
    """


    url_site_map: str
    """The URL to the site map to index."""