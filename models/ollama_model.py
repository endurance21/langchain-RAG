"""Ollama model creation and configuration."""

from langchain_ollama import ChatOllama
from config.settings import OLLAMA_MODEL, OLLAMA_BASE_URL


def create_ollama_model():
    """Create and return an Ollama chat model.
    
    Returns:
        ChatOllama: Configured Ollama chat model
        
    Raises:
        Exception: If model creation fails
    """
    try:
        ollama_model = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.7,  # Controls randomness (0.0 to 1.0)
            num_ctx=4096,  # Context window size
        )
        return ollama_model
    except Exception as e:
        raise Exception(
            f"Error creating Ollama model: {e}. "
            f"Make sure Ollama is running (ollama serve) and model is installed (ollama pull {OLLAMA_MODEL})"
        )

