"""
Query augmentation module for RAG.
Takes a user query, retrieves relevant documents from Pinecone, and creates an augmented system prompt.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Add RAG directory to path for imports
rag_dir = Path(__file__).parent.parent
sys.path.insert(0, str(rag_dir.parent))
from RAG.retrieval.retrieve_from_pinecone import get_relevant_docs

load_dotenv()


def get_augmented_system_prompt(query: str, k: int = 5) -> str:
    """
    Get an augmented system prompt with relevant context from Pinecone.
    
    Args:
        query: The user's query/question
        k: Number of relevant documents to retrieve (default: 5)
        
    Returns:
        str: Augmented system prompt with retrieved context
    """
    # Retrieve relevant documents from Pinecone
    relevant_docs = get_relevant_docs(query, k=k)
    
    # Extract text content from retrieved documents
    context_parts = []
    for doc in relevant_docs:
        # Handle different document formats
        if isinstance(doc, dict):
            # If metadata has page_content, use it
            text = doc.get("metadata", {}).get("page_content", "")
            if not text:
                # Fallback to other possible fields
                text = doc.get("text", "") or doc.get("content", "")
        else:
            # If it's a Document object
            text = getattr(doc, "page_content", "") or getattr(doc, "text", "")
        
        if text:
            context_parts.append(text)
    
    # Combine all context into a single string
    context = "\n\n".join(context_parts)
    
    # Create augmented system prompt
    augmented_prompt = f"""You are a helpful assistant with access to relevant context from a knowledge base.

Use the following context to answer the user's question. If the context doesn't contain enough information to answer the question, say so.

Context from knowledge base:
{context}

User's question: {query}

Instructions:
- Answer the question based on the provided context
- If the context doesn't contain enough information, acknowledge this
- Cite specific parts of the context when relevant
- Be concise and accurate
"""
    
    return augmented_prompt


def get_augmented_prompt_template(query: str, k: int = 5) -> dict:
    """
    Get augmented prompt as a dictionary with system and user messages.
    
    Args:
        query: The user's query/question
        k: Number of relevant documents to retrieve (default: 5)
        
    Returns:
        dict: Dictionary with 'system' and 'user' keys for prompt formatting
    """
    relevant_docs = get_relevant_docs(query, k=k)
    
    # Extract context from documents
    context_parts = []
    for doc in relevant_docs:
        if isinstance(doc, dict):
            text = doc.get("metadata", {}).get("page_content", "") or doc.get("text", "") or doc.get("content", "")
        else:
            text = getattr(doc, "page_content", "") or getattr(doc, "text", "")
        
        if text:
            context_parts.append(text)
    
    context = "\n\n".join(context_parts)
    
    return {
        "system": f"""You are a helpful assistant with access to relevant context from a knowledge base.

Use the following context to answer questions. If the context doesn't contain enough information, say so.

Context from knowledge base:
{context}""",
        "user": query
    }