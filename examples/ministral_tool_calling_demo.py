"""
Ministral Tool Calling Demo

This example demonstrates how to use Ministral (3B or 8B) for tool calling
with llama.cpp via the llama-cpp-python library.

Prerequisites:
1. Download a Ministral GGUF model (e.g., Ministral-8B-Instruct-2410.Q4_K_M.gguf)
2. Install dependencies: pip install llama-cpp-python

Usage:
    python examples/ministral_tool_calling_demo.py --model-path /path/to/model.gguf
"""

import argparse
import json
from llama_cpp import Llama


def get_current_weather(location, unit="celsius"):
    """Mock function to get weather."""
    # In a real app, you would call an API here
    if "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    elif "sf" in location.lower() or "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "16", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


def define_tools():
    """Define available tools in OpenAI format."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]


def run_tool_calling_example(model_path: str, query: str = "What's the weather like in Paris today?"):
    """
    Run a complete tool calling example with Ministral.

    Args:
        model_path: Path to the Ministral GGUF model
        query: User query to process
    """
    print("=" * 80)
    print("Ministral Tool Calling Demo")
    print("=" * 80)
    print(f"\nLoading model from: {model_path}")

    # 1. Setup the model
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,  # Ministral has a large context, 4k is safe for testing
        n_gpu_layers=-1,  # Offload all layers to GPU (set to 0 for CPU only)
        verbose=False,
    )

    print("Model loaded successfully!\n")

    # 2. Define tools
    tools = define_tools()

    # 3. First Pass: Ask the model a question
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query},
    ]

    print(f">>> User: {messages[-1]['content']}\n")

    response = llm.create_chat_completion(
        messages=messages, tools=tools, tool_choice="auto"
    )

    # 4. Handle the Tool Call
    choice = response["choices"][0]
    if choice["finish_reason"] == "tool_calls":
        tool_calls = choice["message"]["tool_calls"]

        # Append the assistant's "tool call" request to history
        messages.append(choice["message"])

        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])

            print(f">>> Model calling tool: {function_name}({arguments})")

            # Execute the function
            if function_name == "get_current_weather":
                function_result = get_current_weather(**arguments)

                print(f">>> Function result: {function_result}\n")

                # Append the result to history
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": function_result,
                    }
                )

        # 5. Second Pass: Get final answer
        final_response = llm.create_chat_completion(messages=messages, tools=tools)

        print(f">>> Final Answer: {final_response['choices'][0]['message']['content']}")
    else:
        # Model didn't make a tool call
        print(f">>> Direct Response: {choice['message']['content']}")
        print("\nNote: The model responded directly without calling a tool.")

    print("\n" + "=" * 80)


def run_interactive_demo(model_path: str):
    """
    Run an interactive tool calling demo.

    Args:
        model_path: Path to the Ministral GGUF model
    """
    print("=" * 80)
    print("Ministral Interactive Tool Calling Demo")
    print("=" * 80)
    print(f"\nLoading model from: {model_path}")

    # Setup the model
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=-1,
        verbose=False,
    )

    print("Model loaded successfully!")
    print("\nType your queries (or 'quit' to exit):")

    # Define tools
    tools = define_tools()

    while True:
        user_input = input("\n>>> You: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not user_input:
            continue

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input},
        ]

        try:
            response = llm.create_chat_completion(
                messages=messages, tools=tools, tool_choice="auto"
            )

            choice = response["choices"][0]
            if choice["finish_reason"] == "tool_calls":
                tool_calls = choice["message"]["tool_calls"]
                messages.append(choice["message"])

                for tool_call in tool_calls:
                    function_name = tool_call["function"]["name"]
                    arguments = json.loads(tool_call["function"]["arguments"])

                    print(f">>> Tool Call: {function_name}({arguments})")

                    if function_name == "get_current_weather":
                        function_result = get_current_weather(**arguments)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "content": function_result,
                            }
                        )

                final_response = llm.create_chat_completion(messages=messages, tools=tools)
                print(f">>> Assistant: {final_response['choices'][0]['message']['content']}")
            else:
                print(f">>> Assistant: {choice['message']['content']}")

        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ministral Tool Calling Demo with llama.cpp"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to Ministral GGUF model file",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What's the weather like in Paris today?",
        help="Query to test (for single query mode)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )

    args = parser.parse_args()

    if not args.model_path:
        print("Error: --model-path is required")
        print("\nExample usage:")
        print("  python examples/ministral_tool_calling_demo.py --model-path /path/to/model.gguf")
        print("  python examples/ministral_tool_calling_demo.py --model-path /path/to/model.gguf --interactive")
        return

    if args.interactive:
        run_interactive_demo(args.model_path)
    else:
        run_tool_calling_example(args.model_path, args.query)


if __name__ == "__main__":
    main()
