from typing import List, Union, cast

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from openai_client import get_completion

MessageParam = Union[
    ChatCompletionUserMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionAssistantMessageParam,
]

def main():
    # Example 1: Simple completion
    messages: List[MessageParam] = [
        cast(ChatCompletionUserMessageParam, {
            "role": "user",
            "content": "What is the capital of France?"
        })
    ]
    
    print("Example 1: Simple completion")
    user_msg = cast(ChatCompletionUserMessageParam, messages[0])
    print("Query:", user_msg["content"])
    completion, metrics = get_completion(messages)
    print("\nResponse:", completion)
    print("\nMetrics:")
    print(f"Prompt tokens: {metrics.prompt_tokens}")
    print(f"Completion tokens: {metrics.completion_tokens}")
    print(f"Total tokens: {metrics.total_tokens}")
    print(f"Total cost: ${metrics.total_cost:.4f}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Multi-turn conversation
    messages: List[MessageParam] = [
        cast(ChatCompletionSystemMessageParam, {
            "role": "system",
            "content": "You are a helpful assistant."
        }),
        cast(ChatCompletionUserMessageParam, {
            "role": "user",
            "content": "What's the best way to learn Python?"
        }),
        cast(ChatCompletionAssistantMessageParam, {
            "role": "assistant",
            "content": "Start with the basics like variables, loops, and functions. "
                      "Practice with small projects."
        }),
        cast(ChatCompletionUserMessageParam, {
            "role": "user",
            "content": "Can you suggest a simple project idea?"
        })
    ]
    
    print("Example 2: Multi-turn conversation")
    print("Messages:")
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        print(f"{role}: {content}")
    
    completion, metrics = get_completion(messages)
    print("\nResponse:", completion)
    print("\nMetrics:")
    print(f"Prompt tokens: {metrics.prompt_tokens}")
    print(f"Completion tokens: {metrics.completion_tokens}")
    print(f"Total tokens: {metrics.total_tokens}")
    print(f"Total cost: ${metrics.total_cost:.4f}")

if __name__ == "__main__":
    main()