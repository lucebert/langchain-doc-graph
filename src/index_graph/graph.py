"""This "graph" simply exposes an endpoint for a user to upload docs to be indexed."""

import asyncio
import os
from typing import List, Optional

import requests
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

from index_graph.configuration import IndexConfiguration
from index_graph.state import IndexState, InputState
from shared import retrieval
from shared.utils import load_pinecone_index


def check_index_config(state: IndexState, *, config: Optional[RunnableConfig] = None) -> dict[str, str]:
    """Check the API key."""
    configuration = IndexConfiguration.from_runnable_config(config)

    if not configuration.api_key:
        raise ValueError("API key is required for document indexing.")
    
    if configuration.api_key != os.getenv("INDEX_API_KEY"):
        raise ValueError("Authentication failed: Invalid API key provided.")
    
    if configuration.retriever_provider != "pinecone":
        raise ValueError("Only Pinecone is currently supported for document indexing due to specific ID prefix requirements.")
    
    return {}

async def get_sitemap_urls(state: IndexState, *, config: Optional[RunnableConfig] = None) -> dict[str, str]:
    """Get the URLs from the sitemap."""
    url = state.url_site_map
    
    headers = {
        "Accept": "application/xml",
        "User-Agent": "Mozilla/5.0 (compatible; LangChainBot/1.0)",
    }
    response = requests.get(url, headers=headers)
    sitemap_content = response.text
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
    urls_to_index = [
        url.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc").text
        for url in root.findall("{http://www.sitemaps.org/schemas/sitemap/0.9}url")
    ]

    print(f"Found {len(urls_to_index)} URLs to index.")

    return {"urls_to_index": urls_to_index}

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
    # Process all URLs in parallel
    chunk_tasks = [index_url(url, config) for url in state.urls_to_index]
    await asyncio.gather(*chunk_tasks)

    return {}


async def index_url(url: str, config: IndexConfiguration) -> List[Document]:
    """Index a web path."""
    loader = WebBaseLoader(
        web_paths=(url,),
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(docs)

    # Delete all existing documents with the same prefix
    index = load_pinecone_index(config.pinecone_index)
    print(f"Deleting {url}#")
    for ids in index.list(prefix=f"{url}#"):
        print(f"Deleting {ids}")
        index.delete(ids=ids)

    # Add the new documents to the index
    # with retrieval.make_retriever(config) as retriever:
        # await retriever.vectorstore.aadd_texts(
        #     texts=[doc.page_content for doc in docs],
        #     metadatas=[doc.metadata for doc in docs],
        #     id_prefix=url,
        # )

    return docs


# Define the graph
builder = StateGraph(IndexState, input=InputState, config_schema=IndexConfiguration)
builder.add_node(check_index_config)
builder.add_node(index_docs)
builder.add_node(get_sitemap_urls)
builder.add_edge(START, "check_index_config")
builder.add_edge("check_index_config", "get_sitemap_urls")
builder.add_edge("get_sitemap_urls", "index_docs")
builder.add_edge("index_docs", END)
# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "IndexGraph"
