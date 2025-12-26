"""Response schemas for structured agent responses using Pydantic."""

from typing import List
from pydantic import Field, BaseModel, field_validator


class AgentResponse(BaseModel):
    """Structured Pydantic response from the agent.
    
    This class provides a validated, structured format for agent responses.
    All responses are automatically validated and can be serialized to JSON/dict.
    
    Example:
        >>> response = AgentResponse(answer="It's sunny!", sources=["htsps://time of lind", https://time of lind2"])
        >>> response.model_dump_json()
        '{"answer":"It\\'s sunny!","sources":["link1", "link2"]}'
    """
    answer: str = Field(
        description="The answer to the user's question",
        min_length=0
    )
    sources: List[str] = Field(
        description="List of source URLs returned by Tavily search results , it should be a list of strings of urls like https://time of lind, https://time of lind2",
        default_factory=list
    )

    @field_validator('answer')
    @classmethod
    def validate_answer(cls, v: str) -> str:
        """Validate that answer is not empty (unless explicitly empty string)."""
        return v.strip() if isinstance(v, str) else str(v)

    def get_answer(self) -> str:
        """Get the answer from the response.
        
        Returns:
            str: The answer to the user's question
        """
        return self.answer

    def get_sources(self) -> List[str]:
        """Get the sources from the response.
        
        Returns:
            List[str]: The sources used to answer the question
        """
        return self.sources
    
    def to_dict(self) -> dict:
        """Convert response to dictionary.
        
        Returns:
            dict: Response as dictionary
        """
        return self.model_dump()
    
    def to_json(self, indent: int = 2) -> str:
        """Convert response to JSON string.
        
        Args:
            indent: JSON indentation level
            
        Returns:
            str: Response as JSON string
        """
        return self.model_dump_json(indent=indent)

