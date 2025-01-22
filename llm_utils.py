from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # Load environment variables from .env file

class Messages:
    """A class to manage chat messages for LLM interactions."""
    
    def __init__(self):
        self._messages: List[Dict[str, str]] = []
    
    def add_system(self, content: str) -> None:
        """Add a system message."""
        self._messages.append({"role": "system", "content": content})
    
    def add_user(self, content: str) -> None:
        """Add a user message."""
        self._messages.append({"role": "user", "content": content})
    
    def add_assistant(self, content: str) -> None:
        """Add an assistant message."""
        self._messages.append({"role": "assistant", "content": content})
    
    def get_messages(self) -> List[Any]:
        """Get the list of messages in OpenAI-compatible format."""
        return self._messages
    
    def clear(self) -> None:
        """Clear all messages."""
        self._messages = []

@dataclass
class CompletionMetrics:
    """Stores token usage and cost information for an OpenAI API call."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    total_cost: float

    @classmethod
    def from_usage(cls, usage: Dict[str, int]) -> 'CompletionMetrics':
        # Cost per 1K tokens for gpt-4o-mini-2024-07-18
        PROMPT_COST_PER_1K = 0.01
        COMPLETION_COST_PER_1K = 0.03
        
        prompt_tokens = usage['prompt_tokens']
        completion_tokens = usage['completion_tokens']
        total_tokens = usage['total_tokens']
        
        # Calculate costs
        prompt_cost = (prompt_tokens / 1000) * PROMPT_COST_PER_1K
        completion_cost = (completion_tokens / 1000) * COMPLETION_COST_PER_1K
        total_cost = prompt_cost + completion_cost
        
        return cls(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            total_cost=total_cost
        )

def get_completion(messages: Messages | List[Dict[str, str]]) -> Tuple[str, CompletionMetrics]:
    """
    Get a completion from OpenAI's API.
    
    Args:
        messages: Either a Messages object or a list of message dictionaries.
                 Example: Messages().add_user("Hello") or
                         [{"role": "user", "content": "Hello"}]
        
    Returns:
        Tuple of (completion_text, CompletionMetrics)
        
    Raises:
        ValueError: If the response content is None
    """
    # Convert Messages object to list if needed
    if isinstance(messages, Messages):
        messages = messages.get_messages()
    client = OpenAI()
    
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=messages,
        temperature=0.7  # Add some variability to responses while keeping them focused
    )
    
    if not response.choices or not response.choices[0].message.content:
        raise ValueError("No completion content received from API")
    
    completion_text = response.choices[0].message.content
    
    if not response.usage:
        raise ValueError("No usage information received from API")
    
    usage_dict = response.usage.model_dump()
    metrics = CompletionMetrics.from_usage(usage_dict)
    
    return completion_text, metrics