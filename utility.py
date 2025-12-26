

from ast import List
from langchain_core.prompts import PromptTemplate
from pydantic import Field, BaseModel


class AgentResponse(BaseModel):
    """
    This class is used to store the response from the agent.
    """
    answer: str = Field(description="The answer to the user's question")
    sources: list[str] = Field(description="The sources used to answer the user's question")

    def __init__(self, answer: str, sources: list[str]):
        self.answer = answer
        self.sources = sources

    def get_answer(self):
        return self.answer

    def get_sources(self):
        return self.sources



def get_prompt_template(system_prompt: str, user_prompt: str) -> str:
    """
    This function is used to get the prompt template for the agent.
    """
    SYSTEM_PROMPT_TEMPLATE = PromptTemplate(
        input_variables=["question"],
        template="""You are a helpful assistant that answers user questions.

    Given the user's question: {question}

    Your task:
    1. Understand what the user is asking
    2. Use available tools if needed to get information
    3. Provide a helpful, friendly answer
    4.dont tell user that you are callilng and tool or system related information, the user is unawar of this total backend proces.
    
    the final answer should be in the following format: {format_instructions}
    Answer in a sweet and friendly manner. Always try to help the user with their question."""
    )
    

