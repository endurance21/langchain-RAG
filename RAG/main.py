"""
RAG (Retrieval Augmented Generation) Class.

This class integrates:
- Document retrieval from Pinecone
- Query augmentation with retrieved context
- LLM model for generating answers
"""

import sys
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

# Configure logging
logger = logging.getLogger(__name__)

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger.info(f"RAG module: Project root set to {project_root}")

try:
    from RAG.augmentation.augment import get_augmented_system_prompt, get_augmented_prompt_template
    logger.info("Successfully imported augmentation functions")
except Exception as e:
    logger.error(f"Failed to import augmentation functions: {e}", exc_info=True)
    raise

try:
    from models.ollama_model import create_ollama_model
    logger.info("Successfully imported create_ollama_model")
except Exception as e:
    logger.error(f"Failed to import create_ollama_model: {e}", exc_info=True)
    raise

load_dotenv()
logger.info("Environment variables loaded")


class RAG:
    """
    RAG class that combines retrieval, augmentation, and generation.
    
    This class:
    1. Retrieves relevant documents from Pinecone based on user query
    2. Augments the prompt with retrieved context
    3. Uses an LLM model to generate answers based on the augmented context
    """
    
    def __init__(self, k: int = 5):
        """
        Initialize the RAG system.
        
        Args:
            k: Number of relevant documents to retrieve from Pinecone (default: 5)
        """
        logger.info(f"Initializing RAG with k={k}")
        self.k = k
        self.llm = None
        try:
            self._initialize_model()
            logger.info("RAG initialization completed successfully")
        except Exception as e:
            logger.error(f"RAG initialization failed: {e}", exc_info=True)
            raise
    
    def _initialize_model(self):
        """Initialize the LLM model."""
        logger.info("Starting model initialization...")
        try:
            logger.info("Calling create_ollama_model()...")
            self.llm = create_ollama_model()
            logger.info(f"Model created successfully. Model type: {type(self.llm)}")
        except Exception as e:
            logger.error(f"Model initialization failed: {e}", exc_info=True)
            raise
    
    def query(self, user_query: str, use_rag: bool = True) -> str:
        """
        Query the RAG system with a user question.
        
        Args:
            user_query: The user's question/query
            use_rag: If True, retrieve context from Pinecone and augment the prompt.
                    If False, use the model without RAG augmentation (default: True)
        
        Returns:
            str: The answer to the user's question
        """
        logger.info(f"Query called: query='{user_query}', use_rag={use_rag}, k={self.k}")
        
        if not self.llm:
            logger.error("Model not initialized")
            raise RuntimeError("Model not initialized. Call _initialize_model() first.")
        
        try:
            if use_rag:
                logger.info("Using RAG mode - retrieving context from Pinecone...")
                # Get augmented prompt with context from Pinecone
                logger.info(f"Calling get_augmented_prompt_template with k={self.k}...")
                augmented_prompt = get_augmented_prompt_template(user_query, k=self.k)
                logger.info(f"Augmented prompt retrieved. System prompt length: {len(augmented_prompt.get('system', ''))}")
                
                # Create prompt template with system and user messages
                logger.info("Creating ChatPromptTemplate...")
                prompt = ChatPromptTemplate.from_messages([
                    ("system", augmented_prompt["system"]),
                    ("user", augmented_prompt["user"])
                ])
            else:
                logger.info("Using LLM-only mode (no RAG)...")
                # Use model without RAG augmentation
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful assistant."),
                    ("user", user_query)
                ])
            
            # Invoke model with prompt
            logger.info("Formatting messages...")
            messages = prompt.format_messages()
            logger.info(f"Messages formatted. Number of messages: {len(messages)}")
            
            logger.info("Invoking LLM...")
            response = self.llm.invoke(messages)
            logger.info(f"LLM response received. Response type: {type(response)}")
            
            # Extract content from response
            logger.info("Extracting content from response...")
            if hasattr(response, 'content'):
                answer = response.content
                logger.info(f"Extracted content from response.content. Length: {len(answer)}")
            elif isinstance(response, str):
                answer = response
                logger.info(f"Response is string. Length: {len(answer)}")
            else:
                answer = str(response)
                logger.info(f"Converted response to string. Length: {len(answer)}")
            
            logger.info("Query completed successfully")
            return answer
            
        except Exception as e:
            logger.error(f"Error in query method: {e}", exc_info=True)
            raise


# Example usage
if __name__ == "__main__":
    # Initialize RAG
    rag = RAG(k=5)  # Retrieve top 5 relevant documents
    
    # Example query
    query = "What are the key security considerations mentioned in the document?"
    
    print(f"\nQuery: {query}\n")
    print("="*60)
    
    # Get response
    answer = rag.query(query)
    
    # Display results
    print("\nAnswer:")
    print("-" * 60)
    print(answer)
    
    print("\n" + "="*60)
