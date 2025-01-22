from openai_client import get_completion

def main():
    # Example 1: Simple completion
    messages = [
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    print("Example 1: Simple completion")
    print("Query:", messages[0]["content"])
    completion, metrics = get_completion(messages)
    print("\nResponse:", completion)
    print("\nMetrics:")
    print(f"Prompt tokens: {metrics.prompt_tokens}")
    print(f"Completion tokens: {metrics.completion_tokens}")
    print(f"Total tokens: {metrics.total_tokens}")
    print(f"Total cost: ${metrics.total_cost:.4f}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Multi-turn conversation
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the best way to learn Python?"},
        {"role": "assistant", "content": "Start with the basics like variables, loops, and functions. Practice with small projects."},
        {"role": "user", "content": "Can you suggest a simple project idea?"}
    ]
    
    print("Example 2: Multi-turn conversation")
    print("Messages:")
    for msg in messages:
        print(f"{msg['role']}: {msg['content']}")
    
    completion, metrics = get_completion(messages)
    print("\nResponse:", completion)
    print("\nMetrics:")
    print(f"Prompt tokens: {metrics.prompt_tokens}")
    print(f"Completion tokens: {metrics.completion_tokens}")
    print(f"Total tokens: {metrics.total_tokens}")
    print(f"Total cost: ${metrics.total_cost:.4f}")

if __name__ == "__main__":
    main()