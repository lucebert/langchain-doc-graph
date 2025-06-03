# LangChainDoc.com - Backend RAG System 🦜🔗

The backend powering [LangChainDoc.com](https://langchaindoc.com) - a RAG system for querying LangChain documentation.

## Overview

This project provides the backend for LangChainDoc.com, enabling querying of LangChain documentation. It uses:

- **LangGraph** for orchestrating the retrieval and response generation
- **Vector database** for storing and retrieving documentation content
- **LLMs** for generating responses with developer insights

## Features

- **Semantic Search**: Find relevant documentation based on meaning
- **Context-Aware Responses**: Responses consider multiple documentation sources

## Supported Providers

This project has been tested with:
- **Vector Database**: Pinecone
- **LLM**: OpenAI

The system is structured to work with other providers, but implementations for alternatives would need to be added.

## Prerequisites
  Before you begin, ensure you have the following API keys for:
        
- [LangSmith](https://smith.langchain.com/settings?_gl=1*1abavxa*_ga*MTExOTQ4NDEzMy4xNzQ4ODcyNzQ3*_ga_47WX3HKKY2*czE3NDg4NzI3NDYkbzEkZzEkdDE3NDg4NzQ3OTAkajMxJGwwJGgw)
- [OpenAI](https://platform.openai.com/signup) (for LLM access)
- [Pinecone](https://www.pinecone.io/start/) (for vector database access)
## Getting Started

1. Copy `.env.example` to `.env`
```bash
cp .env.example .env 
```
2. Add your API keys and configuration to `.env`

3. Install the LangGraph CLI
```bash
pip install --upgrade "langgraph-cli[inmem]"
```

4. Launch LangGraph Server
Start the LangGraph API server locally:

```bash
langgraph dev
```

## Integration with Frontend

This backend system works with the [LangChainDoc Client](https://github.com/lucebert/langchain-doc-client) to provide a complete user experience.

## About the Creator

This project is maintained by [Luc Ebert](https://www.linkedin.com/in/luc-ebert/), a LangChain developer.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)

## Contact

For questions and support, please open an issue in the repository.

