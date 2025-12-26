"""Search-related tools using Tavily."""

import os
from langchain_tavily import TavilySearch


def get_tavily_search_tool():
    """Get Tavily search tool.
    
    Returns:
        TavilySearch: Configured Tavily search tool
        
    Note:
        Requires TAVILY_API_KEY environment variable to be set.
        Get your API key from: https://tavily.com/
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY not found in environment variables. "
            "Please set it in your .env file or export it. "
            "Get your API key from: https://tavily.com/"
        )
    
    return TavilySearch(
        api_key=api_key,
        max_results=3,  # Number of search results to return
    )

