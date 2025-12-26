import os
import logging
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

load_dotenv()

logger.info("Initializing Pinecone client...")
pc = Pinecone()

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
if not PINECONE_INDEX_NAME:
    logger.error("PINECONE_INDEX_NAME environment variable is required")
    raise ValueError("PINECONE_INDEX_NAME environment variable is required")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE")
logger.info(f"Pinecone index: {PINECONE_INDEX_NAME}, namespace: {PINECONE_NAMESPACE}")

# Initialize embedding model (same as used in ingestion)
logger.info("Loading embedding model...")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
logger.info("Embedding model loaded successfully")


def get_relevant_docs(query: str, k: int = 5) -> list[dict]:
    """
    Get relevant documents from Pinecone using semantic search.
    
    Args:
        query: The search query
        k: Number of results to return (default: 5)
        
    Returns:
        list[dict]: List of relevant documents with metadata
    """
    logger.info(f"Getting relevant docs for query: '{query}', k={k}")
    
    try:
        # Get index
        logger.info(f"Connecting to Pinecone index: {PINECONE_INDEX_NAME}")
        index = pc.Index(PINECONE_INDEX_NAME)
        logger.info("Index connection established")
        
        # Encode query to embedding
        logger.info("Encoding query to embedding...")
        query_embedding = embedding_model.encode(query, normalize_embeddings=True).tolist()
        logger.info(f"Query encoded. Embedding dimension: {len(query_embedding)}")
        
        # Query Pinecone using the correct API
        logger.info(f"Querying Pinecone with top_k={k}, namespace={PINECONE_NAMESPACE}...")
        results = index.query(
            vector=query_embedding,
            top_k=k,
            namespace=PINECONE_NAMESPACE,
            include_metadata=True
        )
        logger.info(f"Pinecone query completed. Results type: {type(results)}")
        
        # Extract documents from results (Pinecone returns a QueryResponse dataclass)
        documents = []
        # Access matches as attribute (dataclass), not dictionary key
        if results and hasattr(results, 'matches') and results.matches:
            logger.info(f"Processing {len(results.matches)} matches")
            for match in results.matches:
                # Match is also a dataclass, access attributes directly
                doc = {
                    "id": match.id if hasattr(match, 'id') else None,
                    "score": match.score if hasattr(match, 'score') else 0.0,
                    "metadata": match.metadata if hasattr(match, 'metadata') else {}
                }
                documents.append(doc)
                logger.debug(f"Added document: id={doc['id']}, score={doc.get('score', 0):.4f}")
        else:
            logger.warning("No matches found in Pinecone results")
        
        logger.info(f"Returning {len(documents)} documents")
        return documents
        
    except Exception as e:
        logger.error(f"Error retrieving documents from Pinecone: {e}", exc_info=True)
        raise

    