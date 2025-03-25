"""Manage the configuration of various retrievers.

This module provides functionality to create and manage retrievers for different
vector store backends, specifically Elasticsearch, Pinecone, and MongoDB.
"""

import os
from contextlib import contextmanager, asynccontextmanager
from typing import AsyncGenerator, Tuple
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStoreRetriever

from shared.configuration import BaseConfiguration


## Encoder constructors
def make_text_encoder(model: str) -> Embeddings:
    """Connect to the configured text encoder."""
    provider, model = model.split("/", maxsplit=1)
    match provider:
        case "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model=model)
        case _:
            raise ValueError(f"Unsupported embedding provider: {provider}")


## Retriever constructors
@asynccontextmanager
async def make_pinecone_retriever(
    configuration: BaseConfiguration, embedding_model: Embeddings
) -> AsyncGenerator[Tuple[VectorStoreRetriever, "PineconeVectorStore"], None]:
    """Configure this agent to connect to a specific Pinecone index and return both retriever and vectorstore."""

    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone, ServerlessSpec

    pinecone_client = Pinecone(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT"]
    )

    index_name = os.environ["PINECONE_INDEX_NAME"]
    indexes = pinecone_client.list_indexes().names()

    print("🔎 Index disponibles :", indexes)

    if index_name not in indexes:
        print(f"⚠️ L'index '{index_name}' n'existe pas. Création...")
       # pinecone_client.create_index(name=index_name, dimension=1536, metric="cosine")

        pinecone_client.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",  # or "gcp"
                region="us-east-1" # adapt
            )
        )
        print(f"✅ Index '{index_name}' créé.")

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding_model
    )

    retriever = vectorstore.as_retriever(search_kwargs=configuration.search_kwargs)

    yield retriever, vectorstore

@asynccontextmanager
async def make_retriever(
    config: RunnableConfig,
) -> AsyncGenerator[Tuple[VectorStoreRetriever, object], None]:
    """
    Create a retriever for the agent, based on the current configuration.
    Returns both the retriever and the underlying vectorstore (if available).
    """
    configuration = BaseConfiguration.from_runnable_config(config)
    embedding_model = make_text_encoder(configuration.embedding_model)

    match configuration.retriever_provider:
        case "pinecone":
            async with make_pinecone_retriever(configuration, embedding_model) as (retriever, vectorstore):
                yield retriever, vectorstore

        case _:
            raise ValueError(
                "❌ Unrecognized retriever_provider in configuration. "
                f"Expected one of: {', '.join(BaseConfiguration.__annotations__['retriever_provider'].__args__)}\n"
                f"Got: {configuration.retriever_provider}"
            )
