# LangChain Documentation RAG

A simple RAG (Retrieval Augmented Generation) system for querying LangChain documentation using LangGraph.

[![CI](https://github.com/langchain-ai/rag-research-agent-template/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/langchain-ai/rag-research-agent-template/actions/workflows/unit-tests.yml)

## Overview

This project provides a simple way to query LangChain documentation using a retrieval-based system. It uses:

- LangGraph for orchestrating the retrieval and response generation
- Vector database for storing and retrieving documentation content 
- LLMs for generating natural language responses

## Features

- Document indexing for LangChain documentation
- Natural language querying of documentation content
- Contextual responses based on retrieved documentation

## Getting Started

1. Copy `.env.example` to `.env`
```bash
cp .env.example .env 
```

2. Add your API keys to `.env`:
```
OPENAI_API_KEY=<your-key>
ELASTICSEARCH_URL=<your-url>
ELASTICSEARCH_API_KEY=<your-key>
```

3. Index the documentation:
```python
python index.py
```

4. Start querying the documentation:
```python
python query.py "How do I use LangChain agents?"
```

## Customize

You can customize the:
- Vector store (Elasticsearch, MongoDB, Pinecone)
- Embedding model
- Language model for responses
- System prompts and retrieval parameters

Check the configuration files for available options.

## Development

See the [LangGraph documentation](https://github.com/langchain-ai/langgraph) for more details on extending functionality.

