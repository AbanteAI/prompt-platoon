from llm_utils import Messages, get_completion

def main():
    # Example 1: Simple completion
    messages = Messages()
    messages.add_user("What is the capital of France?")
    
    print("Example 1: Simple completion")
    print("Query: What is the capital of France?")
    completion, metrics = get_completion(messages)
    print("\nResponse:", completion)
    print("\nMetrics:")
    print(f"Prompt tokens: {metrics.prompt_tokens}")
    print(f"Completion tokens: {metrics.completion_tokens}")
    print(f"Total tokens: {metrics.total_tokens}")
    print(f"Total cost: ${metrics.total_cost:.4f}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Multi-turn conversation
    messages = Messages()
    messages.add_system("You are a helpful assistant.")
    messages.add_user("What's the best way to learn Python?")
    messages.add_assistant(
        "Start with the basics like variables, loops, and functions. "
        "Practice with small projects."
    )
    messages.add_user("Can you suggest a simple project idea?")
    
    print("Example 2: Multi-turn conversation")
    print("Messages:")
    for msg in messages.get_messages():
        role = msg["role"]
        content = msg["content"]
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