"""Application settings and configuration."""

# Ollama model configuration
# IMPORTANT: Use full 8b model for proper tool calling!
# 1b models cannot properly format tool calls (see TAVILY_TOOL_ISSUE.md)
OLLAMA_MODEL = "llama3.1"  # Full 8b model - required for tool calling
OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama URL

# Debug mode
DEBUG_MODE = False

