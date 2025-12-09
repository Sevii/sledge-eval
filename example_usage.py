"""Example usage of the sledge-eval framework."""

from pathlib import Path

from sledge_eval.evaluator import Evaluator, TestSuite


def main():
    """Example of loading and working with test suites."""
    # Initialize evaluator (model_client would be your LLM client)
    evaluator = Evaluator(model_client=None)

    # Load test suite from JSON
    test_file = Path("tests/test_data/example_test_suite.json")
    test_suite: TestSuite = evaluator.load_test_suite(test_file)

    print(f"Loaded test suite: {test_suite.name}")
    print(f"Description: {test_suite.description}")
    print(f"Number of tests: {len(test_suite.tests)}\n")

    # Display test cases
    for test in test_suite.tests:
        print(f"Test ID: {test.id}")
        print(f"Voice Command: {test.voice_command}")
        print(f"Expected Tool Calls:")
        for tool_call in test.expected_tool_calls:
            print(f"  - {tool_call.name}({tool_call.arguments})")
        print(f"Tags: {', '.join(test.tags)}")
        print("-" * 50)


if __name__ == "__main__":
    main()
