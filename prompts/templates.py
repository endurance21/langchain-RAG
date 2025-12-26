"""Prompt template utilities."""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from schemas.response import AgentResponse


# System prompt template for the agent
SYSTEM_PROMPT_TEMPLATE = """
You are a helpful assistant.

When you use web search, you MUST cite sources as URLs returned by Tavily.

Rules:
- sources must contain ONLY URLs from Tavily results
- do not invent or modify URLs
- no markdown, no extra keys, no text outside JSON

IMPORTANT:
- do not tell user that you are calling a tool or system related information, the user is unaware of this total backend process.
- do not tell user that you are using a tool or system related information, the user is unaware of this total backend process.
- always show the sources you used for the answers that you know from the internet or if you used tool like tavily or other tools, you must show the sources you used for the answers.
The final answer should be in the following format: {format_instructions}

""".strip()

PROMPT_TEMPLATE="""
You are a helpful assistant.
you must answer the user's question :  {input}
When you use web search, you MUST cite sources as URLs returned by Tavily.

Rules:
- sources must contain ONLY URLs from Tavily results
- do not invent or modify URLs
- no markdown, no extra keys, no text outside JSON

IMPORTANT:
- do not tell user that you are calling a tool or system related information, the user is unaware of this total backend process.
- do not tell user that you are using a tool or system related information, the user is unaware of this total backend process.
- always show the sources you used for the answers that you know from the internet or if you used tool like tavily or other tools, you must show the sources you used for the answers.
The final answer should be in the following format: {format_instructions}

"""

def get_full_prompt_template() -> str:
    """Get prompt template for the agent.
    
    Returns:
        str: Prompt template string
    """
    output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
    format_instructions = output_parser.get_format_instructions()
    return PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["input", "format_instructions"]
    ).partial(format_instructions=format_instructions)


def get_system_prompt() -> str:
    """Get formatted system prompt with format instructions.
    
    Returns:
        str: Formatted system prompt string with Pydantic format instructions
    """
    # Create PydanticOutputParser to get format instructions
    output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
    format_instructions = output_parser.get_format_instructions()
    
    # Create PromptTemplate and format it with the instructions
    prompt_template = PromptTemplate(
        template=SYSTEM_PROMPT_TEMPLATE,
        input_variables=["format_instructions"]
    )
    
    # Format the template with the actual format instructions
    return prompt_template.format(format_instructions=format_instructions)


def get_prompt_template(system_prompt: str, user_prompt: str, format_instructions: str = "") -> PromptTemplate:
    """Get a prompt template for the agent.
    
    Args:
        system_prompt: The system prompt to use
        user_prompt: The user prompt/question
        format_instructions: Optional format instructions for structured output
        
    Returns:
        PromptTemplate: Configured prompt template
    """
    template_str = """You are a helpful assistant that answers user questions.

Given the user's question: {question}

Your task:
1. Understand what the user is asking
2. Use available tools if needed to get information
3. Provide a helpful, friendly answer
4. Don't tell user that you are calling a tool or system related information, the user is unaware of this total backend process.
"""
    
    if format_instructions:
        template_str += f"\nThe final answer should be in the following format: {format_instructions}\n"
    
    template_str += "Answer in a sweet and friendly manner. Always try to help the user with their question."
    
    return PromptTemplate(
        input_variables=["question"],
        template=template_str
    )

