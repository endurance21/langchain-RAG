"""Agent factory for creating and using agents with Pydantic structured output."""

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from models.ollama_model import create_ollama_model
from config.settings import DEBUG_MODE
from tools import get_tavily_search_tool
from schemas.response import AgentResponse
from prompts.templates import get_system_prompt


def create_agent_instance():
    """Create and return an agent instance with structured output formatting.
    
    Uses LangChain's native structured output feature. For Ollama models,
    LangChain uses ToolStrategy to achieve structured output.
    
    Returns:
        CompiledStateGraph: Configured agent instance with structured output
    """
    llm = create_ollama_model()
    
    # Get all available tools
    tools = []
    
    # Add Tavily search tool if API key is available
    try:
        tavily_tool = get_tavily_search_tool()
        tools.append(tavily_tool)
    except ValueError:
        # Tavily API key not set, skip adding the tool
        pass
    
    # Get formatted system prompt with format instructions
    system_prompt = get_system_prompt()
    
    # Create agent with structured output using ToolStrategy
    # ToolStrategy is used for models that don't support native structured output (like Ollama)
    agent = create_agent(
        model=llm,
        tools=tools,
        debug=DEBUG_MODE,
        system_prompt=system_prompt,
        response_format=ToolStrategy(AgentResponse),  # LangChain handles structured output
    )
    
    return agent


def get_agent_response(agent, user_query: str) -> AgentResponse:
    """Get structured Pydantic response from agent for a user query.
    
    Uses LangChain's native structured output feature. The structured response
    is automatically validated and returned in the 'structured_response' key.
    
    Args:
        agent: The agent instance
        user_query: The user's question/query
        
    Returns:
        AgentResponse: Validated Pydantic response with answer and sources
    """
    content = ""
    result = agent.invoke({"messages": [{"role": "user", "content": user_query}]})
    if isinstance(result, dict) and "messages" in result:
        last_message = result["messages"][-1]
        if hasattr(last_message, "content"):
            content = last_message.content

    structed_llm = create_ollama_model().with_structured_output(AgentResponse)

    result = structed_llm.invoke(f"Answer the user's question: {content}")

    print(result)

    return result