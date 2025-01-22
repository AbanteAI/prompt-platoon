from dataclasses import dataclass
from typing import List, Tuple
from openai import OpenAI
from openai.types.chat import ChatCompletion
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

@dataclass
class CompletionMetrics:
    """Stores token usage and cost information for an OpenAI API call."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    total_cost: float

    @classmethod
    def from_usage(cls, usage: ChatCompletion.Usage) -> 'CompletionMetrics':
        # Cost per 1K tokens for gpt-4o-mini-2024-07-18
        PROMPT_COST_PER_1K = 0.01
        COMPLETION_COST_PER_1K = 0.03
        
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens
        
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

def get_completion(messages: List[dict]) -> Tuple[str, CompletionMetrics]:
    """
    Get a completion from OpenAI's API.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
                 Example: [{"role": "user", "content": "Hello"}]
        
    Returns:
        Tuple of (completion_text, CompletionMetrics)
    """
    client = OpenAI()
    
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=messages,
        temperature=0.7  # Add some variability to responses while keeping them focused
    )
    
    completion_text = response.choices[0].message.content
    metrics = CompletionMetrics.from_usage(response.usage)
    
    return completion_text, metrics