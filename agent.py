import os
import warnings
from dotenv import load_dotenv
from agent.agent_factory import create_agent_instance, get_agent_response
from schemas.response import AgentResponse

# Suppress warnings from langchain_tavily library
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_tavily")

# Load environment variables from .env file
load_dotenv()


def main():
    # Create agent instance
    agent = create_agent_instance()

    user_query = "is hyderabad new it hub in india ?"

    # Get structured response from agent
    response = get_agent_response(agent, user_query)

    # Display the response
    print("\n" + "="*50)
    print("Answer:")
    print("="*50)
    print(response.answer)
    
    if response.sources:
        print("\n" + "="*50)
        print("Sources:")
        print("="*50)
        for source in response.sources:
            print(f"  - {source}")
    
    print("\n" + "="*50)
    print("JSON Output:")
    print("="*50)
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()