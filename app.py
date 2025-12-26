"""
FastAPI application for RAG (Retrieval Augmented Generation) service.
"""

import sys
import logging
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logger.info(f"Project root: {project_root}")
logger.info("Importing RAG class...")

try:
    from RAG.main import RAG
    logger.info("RAG class imported successfully")
except Exception as e:
    logger.error(f"Failed to import RAG class: {e}", exc_info=True)
    raise

# Initialize FastAPI app
app = FastAPI(
    title="RAG API",
    description="Retrieval Augmented Generation API for querying documents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG instance (singleton)
rag_instance: Optional[RAG] = None


def get_rag_instance() -> RAG:
    """Get or create RAG instance (singleton pattern)."""
    global rag_instance
    if rag_instance is None:
        logger.info("Creating new RAG instance...")
        try:
            rag_instance = RAG(k=5)  # Retrieve top 5 documents
            logger.info("RAG instance created successfully")
        except Exception as e:
            logger.error(f"Failed to create RAG instance: {e}", exc_info=True)
            raise
    else:
        logger.debug("Using existing RAG instance")
    return rag_instance


# Request/Response models
class QueryRequest(BaseModel):
    """Request model for RAG query."""
    query: str = Field(..., description="The user's question/query", min_length=1)
    use_rag: bool = Field(True, description="Whether to use RAG (retrieve from Pinecone) or just LLM")
    k: Optional[int] = Field(None, description="Number of documents to retrieve (overrides default if provided)")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the key security considerations mentioned in the document?",
                "use_rag": True,
                "k": 5
            }
        }


class QueryResponse(BaseModel):
    """Response model for RAG query."""
    answer: str = Field(..., description="The answer to the user's question")
    query: str = Field(..., description="The original query")
    use_rag: bool = Field(..., description="Whether RAG was used")

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Based on the document, the key security considerations include...",
                "query": "What are the key security considerations mentioned in the document?",
                "use_rag": True
            }
        }


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RAG API is running",
        "version": "1.0.0",
        "endpoints": {
            "/query": "POST - Query the RAG system",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    logger.info("Health check endpoint called")
    try:
        logger.info("Getting RAG instance for health check...")
        rag = get_rag_instance()
        logger.info(f"RAG instance obtained. LLM initialized: {rag.llm is not None}")
        return {
            "status": "healthy",
            "rag_initialized": rag.llm is not None
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system with a user question.
    
    - **query**: The user's question
    - **use_rag**: Whether to use RAG (retrieve from Pinecone) or just LLM
    - **k**: Number of documents to retrieve (optional, overrides default)
    """
    logger.info(f"Query endpoint called with query: '{request.query}', use_rag: {request.use_rag}, k: {request.k}")
    try:
        logger.info("Getting RAG instance...")
        rag = get_rag_instance()
        logger.info("RAG instance obtained")
        
        # Override k if provided
        if request.k is not None:
            logger.info(f"Overriding k from {rag.k} to {request.k}")
            rag.k = request.k
        
        # Query the RAG system
        logger.info(f"Calling rag.query() with use_rag={request.use_rag}...")
        answer = rag.query(request.query, use_rag=request.use_rag)
        logger.info(f"Query completed. Answer length: {len(answer) if answer else 0} characters")
        
        response = QueryResponse(
            answer=answer,
            query=request.query,
            use_rag=request.use_rag
        )
        logger.info("Response created successfully")
        return response
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

