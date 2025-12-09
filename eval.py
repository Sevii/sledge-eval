#!/usr/bin/env python3
"""
Sledge Eval CLI

Entry point for running voice command evaluations with various models.

Usage:
    uv run eval.py --model-path /path/to/model.gguf
    uv run eval.py --model-path /path/to/model.gguf --mode suite
    uv run eval.py --model-path /path/to/model.gguf --mode custom
"""

import argparse
from pathlib import Path

from src.sledge_eval import MinistralEvaluator, VoiceCommandTest, ToolCall


def run_single_test(model_path: str):
    """Run a single test evaluation."""
    print("=" * 80)
    print("Single Test Evaluation")
    print("=" * 80)

    # Initialize the evaluator
    print(f"\nInitializing MinistralEvaluator with model: {model_path}")
    evaluator = MinistralEvaluator(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=-1,  # Use GPU if available
        verbose=False,
    )
    print("Evaluator initialized!\n")

    # Create a test case
    test = VoiceCommandTest(
        id="eval_001",
        voice_command="Turn on the living room lights",
        expected_tool_calls=[
            ToolCall(
                name="control_lights",
                arguments={"room": "living room", "action": "turn_on"},
            )
        ],
        description="Simple light control command",
        tags=["lights", "smart_home"],
    )

    print(f"Test ID: {test.id}")
    print(f"Voice Command: '{test.voice_command}'")
    print(f"Expected Tool Calls: {test.expected_tool_calls}\n")

    # Evaluate the test
    print("Evaluating...")
    result = evaluator.evaluate_test(test)

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Test ID: {result.test_id}")
    print(f"Status: {'PASS ✓' if result.passed else 'FAIL ✗'}")
    print(f"\nExpected Tool Calls:")
    for tc in result.expected_tool_calls:
        print(f"  - {tc.name}({tc.arguments})")
    print(f"\nPredicted Tool Calls:")
    for tc in result.predicted_tool_calls:
        print(f"  - {tc.name}({tc.arguments})")

    if result.error:
        print(f"\nError: {result.error}")

    print("=" * 80)


def run_test_suite(model_path: str, test_file: str = None):
    """Run a full test suite evaluation."""
    print("=" * 80)
    print("Test Suite Evaluation")
    print("=" * 80)

    # Initialize the evaluator
    print(f"\nInitializing MinistralEvaluator with model: {model_path}")
    evaluator = MinistralEvaluator(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=-1,
        verbose=False,
    )
    print("Evaluator initialized!\n")

    # Load test suite
    if test_file is None:
        test_file = Path("tests/test_data/example_test_suite.json")

    print(f"Loading test suite from: {test_file}")
    test_suite = evaluator.load_test_suite(Path(test_file))
    print(f"Loaded test suite: {test_suite.name}")
    print(f"Description: {test_suite.description}")
    print(f"Number of tests: {len(test_suite.tests)}\n")

    # Evaluate the suite
    print("Running evaluation...\n")
    results = evaluator.evaluate_suite(test_suite)

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    passed_count = sum(1 for r in results if r.passed)
    total_count = len(results)
    pass_rate = (passed_count / total_count * 100) if total_count > 0 else 0

    print(f"\nOverall: {passed_count}/{total_count} tests passed ({pass_rate:.1f}%)\n")

    # Detailed results
    for result in results:
        status = "PASS ✓" if result.passed else "FAIL ✗"
        print(f"{status} {result.test_id}")

        if not result.passed:
            print(f"  Expected:")
            for tc in result.expected_tool_calls:
                print(f"    - {tc.name}({tc.arguments})")
            print(f"  Predicted:")
            for tc in result.predicted_tool_calls:
                print(f"    - {tc.name}({tc.arguments})")
            if result.error:
                print(f"  Error: {result.error}")
            print()

    print("=" * 80)


def run_custom_tools(model_path: str):
    """Run evaluation with custom tool definitions."""
    print("=" * 80)
    print("Custom Tools Evaluation")
    print("=" * 80)

    # Define custom tools
    custom_tools = [
        {
            "type": "function",
            "function": {
                "name": "send_email",
                "description": "Send an email to a recipient",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {
                            "type": "string",
                            "description": "Email recipient",
                        },
                        "subject": {
                            "type": "string",
                            "description": "Email subject",
                        },
                        "body": {
                            "type": "string",
                            "description": "Email body",
                        },
                    },
                    "required": ["to", "subject"],
                },
            },
        }
    ]

    # Initialize evaluator with custom tools
    print(f"\nInitializing MinistralEvaluator with custom tools")
    evaluator = MinistralEvaluator(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=-1,
        verbose=False,
        available_tools=custom_tools,
    )
    print("Evaluator initialized with custom tools!\n")

    # Create a test with the custom tool
    test = VoiceCommandTest(
        id="custom_001",
        voice_command="Send an email to john@example.com with subject 'Meeting Tomorrow'",
        expected_tool_calls=[
            ToolCall(
                name="send_email",
                arguments={
                    "to": "john@example.com",
                    "subject": "Meeting Tomorrow",
                },
            )
        ],
    )

    print(f"Voice Command: '{test.voice_command}'")
    print(f"Expected Tool: send_email\n")

    # Evaluate
    print("Evaluating...")
    result = evaluator.evaluate_test(test)

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Status: {'PASS ✓' if result.passed else 'FAIL ✗'}")
    print(f"\nPredicted Tool Calls:")
    for tc in result.predicted_tool_calls:
        print(f"  - {tc.name}({tc.arguments})")
    print("=" * 80)


def main():
    """Main entry point for sledge-eval CLI."""
    parser = argparse.ArgumentParser(
        description="Sledge Eval - Voice command to tool call evaluation framework",
        prog="uv run eval.py"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to Ministral GGUF model file",
    )
    parser.add_argument(
        "--test-suite",
        type=str,
        help="Path to test suite JSON file (optional)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "suite", "custom"],
        default="single",
        help="Evaluation mode: single test, test suite, or custom tools (default: single)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Validate model path exists
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found: {args.model_path}")
        return 1

    try:
        if args.mode == "single":
            run_single_test(args.model_path)
        elif args.mode == "suite":
            run_test_suite(args.model_path, args.test_suite)
        elif args.mode == "custom":
            run_custom_tools(args.model_path)
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())