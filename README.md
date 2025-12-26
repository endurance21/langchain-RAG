# RAG Agent Development

A Retrieval Augmented Generation (RAG) system built with LangChain, Ollama, and Pinecone. This project provides both a web search agent and a RAG system for querying documents stored in Pinecone.

## Features

- 🤖 **RAG System**: Query documents using retrieval-augmented generation
- 🔍 **Web Search Agent**: Agent with Tavily search integration
- 🚀 **FastAPI API**: RESTful API for querying the RAG system
- 📚 **Pinecone Integration**: Vector database for document storage and retrieval
- 🦙 **Ollama Support**: Local LLM inference using Ollama

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) installed and running
- [uv](https://github.com/astral-sh/uv) package manager
- Pinecone account and API key
- (Optional) Tavily API key for web search

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd agent_development
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Set up Ollama**
   ```bash
   # Start Ollama (if not already running)
   ollama serve
   
   # Pull the required model
   ollama pull llama3.1
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Set up Pinecone**
   - Create a Pinecone account at [https://app.pinecone.io/](https://app.pinecone.io/)
   - Create an index with dimension 384 (for all-MiniLM-L6-v2 embeddings)
   - Add your API key and index name to `.env`

6. **Ingest documents to Pinecone** (Optional)
   ```bash
   uv run python RAG/ingestion/ingest_to_pinecone.py
   ```

## Quick Start

### Using the RAG API

1. **Start the FastAPI server**
   ```bash
   uv run python app.py
   ```

2. **Query the RAG system**
   ```bash
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What are the key security considerations mentioned in the document?"
     }'
   ```

3. **Access API documentation**
   - Interactive docs: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc

### Using the Web Search Agent

```bash
uv run python agent.py
```

## API Endpoints

### POST `/query`
Query the RAG system with a question.

**Request:**
```json
{
  "query": "Your question here",
  "use_rag": true,
  "k": 5
}
```

**Response:**
```json
{
  "answer": "The answer to your question...",
  "query": "Your question here",
  "use_rag": true
}
```

**Parameters:**
- `query` (required): Your question
- `use_rag` (optional, default: true): Whether to use RAG or just LLM
- `k` (optional): Number of documents to retrieve from Pinecone

### GET `/health`
Check if the API is running and RAG is initialized.

### GET `/`
Get API information and available endpoints.

## Project Structure

```
.
├── RAG/                    # RAG system components
│   ├── main.py            # RAG class and main logic
│   ├── ingestion/         # Document ingestion to Pinecone
│   ├── retrieval/         # Document retrieval from Pinecone
│   └── augmentation/      # Query augmentation with context
├── agent/                  # Agent factory and logic
├── models/                 # LLM model configuration
├── tools/                  # Agent tools (search, etc.)
├── schemas/                # Pydantic response schemas
├── prompts/                # Prompt templates
├── config/                 # Configuration settings
├── app.py                  # FastAPI application
└── agent.py                # Web search agent script
```

## Environment Variables

See `.env.example` for all required environment variables:

- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_INDEX_NAME`: Name of your Pinecone index
- `PINECONE_NAMESPACE`: Namespace in your Pinecone index
- `TAVILY_API_KEY`: (Optional) Tavily API key for web search
- `OLLAMA_BASE_URL`: (Optional) Ollama server URL (default: http://localhost:11434)

## Example Usage

### Query with RAG (retrieves from Pinecone)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the security considerations?",
    "use_rag": true,
    "k": 5
  }'
```

### Query without RAG (just LLM)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is artificial intelligence?",
    "use_rag": false
  }'
```

## Development

### Running Tests
```bash
uv run python -m pytest
```

### Code Formatting
```bash
uv run ruff format .
```

### Type Checking
```bash
uv run mypy .
```

## Troubleshooting

### Ollama Connection Issues
- Make sure Ollama is running: `ollama serve`
- Verify model is installed: `ollama list`
- Check `OLLAMA_BASE_URL` in `.env`

### Pinecone Connection Issues
- Verify your API key is correct
- Check that the index exists and has the correct dimension (384)
- Ensure the namespace exists in your index

### Import Errors
- Make sure all dependencies are installed: `uv sync`
- Check that you're using Python 3.11+

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

