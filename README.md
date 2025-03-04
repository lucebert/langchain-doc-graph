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

## Getting Started

1. Copy `.env.example` to `.env`
```bash
cp .env.example .env 
```

2. Add your API keys and configuration to `.env`

3. Running with LangGraph Studio
- **Mac users**: Use LangGraph Studio directly
- **Windows/Linux users**: Follow [this tutorial](https://langchain-ai.github.io/langgraph/tutorials/langgraph-platform/local-server/#langgraph-studio-web-ui) to set up LangGraph Studio

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

