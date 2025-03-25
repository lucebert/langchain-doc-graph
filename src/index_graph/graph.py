import asyncio
import os
import logging
from typing import List, Optional
from datetime import datetime
from pathlib import Path
import gc
from pinecone import Index

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

# Configure logging for errors and status
LOG_PATH = Path("indexing_errors.log")
logging.basicConfig(
    filename=LOG_PATH,
    filemode="a",
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)

def check_index_config(state: IndexState, *, config: Optional[RunnableConfig] = None) -> dict[str, str]:
    """Validate API key and supported retriever provider."""
    configuration = IndexConfiguration.from_runnable_config(config)

    if not configuration.api_key:
        raise ValueError("API key is required for document indexing.")

    if configuration.api_key != os.getenv("INDEX_API_KEY"):
        raise ValueError("Authentication failed: Invalid API key provided.")

    if configuration.retriever_provider != "pinecone":
        raise ValueError("Only Pinecone is currently supported for document indexing due to specific ID prefix requirements.")

    return {}

async def get_sitemap_urls(state: IndexState, *, config: Optional[RunnableConfig] = None) -> dict[str, str]:
    """Fetch all URLs from a sitemap (XML format)."""
    url = state.url_site_map

    headers = {
        "Accept": "application/xml",
        "User-Agent": "Mozilla/5.0 (compatible; LangChainBot/1.0)",
    }
    response = requests.get(url, headers=headers)
    sitemap_content = response.text

    import xml.etree.ElementTree as ET
    root = ET.fromstring(sitemap_content)
    urls_to_index = [
        url.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc").text
        for url in root.findall("{http://www.sitemaps.org/schemas/sitemap/0.9}url")
    ]

    print(f"Found {len(urls_to_index)} URLs to index.")
    return {"urls_to_index": urls_to_index}

async def index_docs(state: IndexState, *, config: Optional[RunnableConfig] = None) -> dict[str, str]:
    """Index documents from all URLs in batches of 100, without concurrency limitation."""
    # Load Pinecone index once
    index_name = os.environ["PINECONE_INDEX_NAME"]
    index = load_pinecone_index(index_name)

    success_count = 0
    fail_count = 0

    async def safe_index_url(url: str) -> None:
        nonlocal success_count, fail_count
        try:
            await index_url(url, config=config, index=index)
            success_count += 1
        except Exception as e:
            logging.error(f"Failed indexing {url}: {e}")
            with open("failed_urls.txt", "a") as f:
                f.write(f"{url}\n")
            fail_count += 1
        finally:
            gc.collect()

    # Process URLs in batches of 100
    batch_size = 100
    total = len(state.urls_to_index)
    for i in range(0, total, batch_size):
        current_batch = state.urls_to_index[i:i + batch_size]
        print(f"🔄 Processing batch {i // batch_size + 1} / {(total + batch_size - 1) // batch_size}")
        tasks = [safe_index_url(url) for url in current_batch]
        await asyncio.gather(*tasks, return_exceptions=True)

    print(f"Indexed: {success_count} | Failed: {fail_count}")
    return {
        "success_count": str(success_count),
        "fail_count": str(fail_count),
    }


async def index_url(url: str, config: IndexConfiguration, index:Index, retry: int = 1) -> List[Document]:
    """Delete old chunks and re-index content from a given URL."""
    try:
        logging.info(f"Indexing: {url}")
        loader = WebBaseLoader(web_paths=(url,))
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(docs)

        now_str = datetime.utcnow().isoformat()
        for doc in docs:
            doc.metadata["source_url"] = url
            doc.metadata["last_indexed_at"] = now_str

        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        chunk_ids = [f"{url}--chunk{i}" for i in range(len(texts))]

        
        print(f"Checking for existing chunks at prefix: {url}")
        existing_urls = list(index.list(prefix=f"{url}"))

        if existing_urls:
            print(f"Deleted old chunks ({len(existing_urls)}) for {url}")
            index.delete(ids=existing_urls)
        else:
            print(f"No existing chunks found for {url}")

        async with retrieval.make_retriever(config) as (_, vectorstore):
            if hasattr(vectorstore, "aadd_texts"):
                await vectorstore.aadd_texts(
                    texts=texts,
                    metadatas=metadatas,
                    ids=chunk_ids
                )
            else:
                for i, doc in enumerate(docs):
                    doc.metadata["id"] = chunk_ids[i]
                await vectorstore.aadd_documents(docs)

        logging.info(f"Successfully indexed {url}")
        return docs

    except Exception as e:
        if retry > 0:
            logging.warning(f"⚠️ Retry {url} after error: {e}")
            await asyncio.sleep(1)
            return await index_url(url, config,index, retry=retry - 1)
        else:
            logging.error(f"Final failure for {url}: {e}")
            return []

# Define the graph structure
builder = StateGraph(IndexState, input=InputState, config_schema=IndexConfiguration)
builder.add_node(check_index_config)
builder.add_node(index_docs)
builder.add_node(get_sitemap_urls)
builder.add_edge(START, "check_index_config")
builder.add_edge("check_index_config", "get_sitemap_urls")
builder.add_edge("get_sitemap_urls", "index_docs")
builder.add_edge("index_docs", END)

# Compile the state graph for execution
graph = builder.compile()
graph.name = "IndexGraph"
