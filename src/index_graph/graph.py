"""This "graph" simply exposes an endpoint for a user to upload docs to be indexed."""

import asyncio
from typing import List, Optional

import requests
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

from index_graph.configuration import IndexConfiguration
from index_graph.state import IndexState
from shared import retrieval


async def index_docs(
    state: IndexState, *, config: Optional[RunnableConfig] = None
) -> dict[str, str]:
    """Asynchronously index documents in the given state using the configured retriever.

    This function takes the documents from the state, ensures they have a user ID,
    adds them to the retriever's index, and then signals for the documents to be
    deleted from the state.

    If docs are not provided in the state, they will be loaded
    from the configuration.docs_file JSON file.

    Args:
        state (IndexState): The current state containing documents and retriever.
        config (Optional[RunnableConfig]): Configuration for the indexing process.r
    """
    if not config:
        raise ValueError("Configuration required to run index_docs.")

    url = state.url_site_map

    # Load and parse the sitemap

    headers = {
        "Accept": "application/xml",
        "User-Agent": "Mozilla/5.0 (compatible; LangChainBot/1.0)",
    }
    response = requests.get(url, headers=headers)
    sitemap_content = response.text

    # Extract URLs from sitemap (assuming XML format)
    import xml.etree.ElementTree as ET

    root = ET.fromstring(sitemap_content)
    # Extract all URLs, removing frequency and other metadata
    urls = [
        url.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc").text
        for url in root.findall("{http://www.sitemaps.org/schemas/sitemap/0.9}url")
    ]

    # Process each URL
    docs = []

    # Convert synchronous function to async
    async def async_get_web_chunks(url):
        return await asyncio.get_event_loop().run_in_executor(
            None, get_web_chuncks, url
        )

    # Process all URLs in parallel
    chunk_tasks = [async_get_web_chunks(url) for url in urls]
    chunks_list = await asyncio.gather(*chunk_tasks)

    # Flatten the list of lists into single docs list
    docs = [doc for chunks in chunks_list for doc in chunks]

    print(f"Indexing {len(docs)} documents from {url}.")

    with retrieval.make_retriever(config) as retriever:
        batch_size = min(
            500, max(1, len(docs))
        )  # Target 500 docs per batch, but handle smaller doc counts
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            await retriever.aadd_documents(batch)
            print(
                f"Indexed batch {i//batch_size + 1} of {(len(docs) + batch_size - 1)//batch_size}"
            )

    return {"docs": docs}


def get_web_chuncks(path: str) -> List[Document]:
    """Index a web path."""
    loader = WebBaseLoader(
        web_paths=(path,),
    )
    docs = loader.load()

    print(f"Loaded {len(docs)} documents from {path}.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    docs = text_splitter.split_documents(docs)

    print(f"Split {len(docs)} documents into chunks.")
    return docs


# Define the graph
builder = StateGraph(IndexState, config_schema=IndexConfiguration)
builder.add_node(index_docs)
builder.add_edge(START, "index_docs")
builder.add_edge("index_docs", END)
# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "IndexGraph"
