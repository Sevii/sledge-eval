#!/usr/bin/env python3
"""
Demo script showing how to use the TextServerEvaluator for text-based evaluations.

This example demonstrates:
1. Creating a letter counting test suite
2. Running evaluations against a local llama-server
3. Generating comprehensive reports

Usage:
    python examples/text_evaluator_demo.py --server-url http://localhost:8080
"""

import argparse
import json
from pathlib import Path

from sledge_eval import (
    TextServerEvaluator,
    TextEvaluationTest,
    TextEvaluationSuite,
)


def create_letter_counting_suite() -> TextEvaluationSuite:
    """Create a comprehensive letter counting test suite."""
    tests = [
        TextEvaluationTest(
            id="letter_count_001",
            question="How many times does the letter 'r' appear in the word 'strawberry'?",
            expected_answer="3",
            description="Count occurrences of letter 'r' in 'strawberry'",
            tags=["letter_counting", "basic"],
            evaluation_type="letter_count"
        ),
        TextEvaluationTest(
            id="letter_count_002",
            question="How many times does the letter 'l' appear in the word 'parallel'?",
            expected_answer="3",
            description="Count occurrences of letter 'l' in 'parallel'",
            tags=["letter_counting", "basic"],
            evaluation_type="letter_count"
        ),
        TextEvaluationTest(
            id="letter_count_003",
            question="How many times does the letter 'e' appear in the word 'development'?",
            expected_answer="4",
            description="Count occurrences of letter 'e' in 'development'",
            tags=["letter_counting", "medium"],
            evaluation_type="letter_count"
        ),
        TextEvaluationTest(
            id="letter_count_004",
            question="How many times does the letter 's' appear in the word 'assessment'?",
            expected_answer="4",
            description="Count occurrences of letter 's' in 'assessment'",
            tags=["letter_counting", "medium"],
            evaluation_type="letter_count"
        ),
        TextEvaluationTest(
            id="letter_count_005",
            question="How many times does the letter 'i' appear in the word 'Mississippi'?",
            expected_answer="4",
            description="Count occurrences of letter 'i' in 'Mississippi'",
            tags=["letter_counting", "hard"],
            evaluation_type="letter_count"
        )
    ]
    
    return TextEvaluationSuite(
        name="Letter Counting Evaluation",
        description="Comprehensive test suite for evaluating letter counting abilities in language models",
        tests=tests
    )


def save_test_suite(suite: TextEvaluationSuite, output_path: Path):
    """Save test suite to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(suite.model_dump(), f, indent=2)
    
    print(f"✅ Test suite saved to: {output_path}")


def run_evaluation(server_url: str, test_suite: TextEvaluationSuite):
    """Run text evaluation against a server."""
    print(f"🚀 Starting text evaluation against: {server_url}")
    
    # Initialize evaluator
    evaluator = TextServerEvaluator(server_url=server_url)
    
    # Health check
    print("🔍 Checking server health...")
    if not evaluator.health_check():
        print("❌ Server health check failed. Make sure llama-server is running.")
        return
    
    print("✅ Server is healthy")
    
    # Run evaluation
    print(f"📝 Running {len(test_suite.tests)} text evaluation tests...")
    results = evaluator.evaluate_text_suite(test_suite)
    
    # Create report
    model_name = "llama-server"  # You can detect this from the server if needed
    report = evaluator.create_text_evaluation_report(
        results=results,
        model_name=model_name,
        test_suite_name=test_suite.name,
        evaluation_mode="text_evaluation"
    )
    
    # Save report
    base_path = Path(".")
    file_paths = report.save_to_file(base_path)
    
    print("\n📊 Evaluation Results:")
    print(f"   • Total Tests: {report.total_tests}")
    print(f"   • Passed: {report.passed_tests}")
    print(f"   • Failed: {report.failed_tests}")
    print(f"   • Pass Rate: {report.pass_rate:.1f}%")
    print(f"   • Total Time: {report.total_evaluation_time_ms:.1f}ms")
    
    print(f"\n📄 Reports saved:")
    print(f"   • JSON: {file_paths['json']}")
    print(f"   • Markdown: {file_paths['markdown']}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Text Evaluator Demo")
    parser.add_argument(
        "--server-url",
        default="http://localhost:8080",
        help="URL of the llama-server instance"
    )
    parser.add_argument(
        "--save-suite",
        action="store_true",
        help="Save test suite to JSON file"
    )
    parser.add_argument(
        "--suite-path",
        default="tests/test_data/letter_counting_suite.json",
        help="Path to save/load test suite"
    )
    
    args = parser.parse_args()
    
    # Create test suite
    test_suite = create_letter_counting_suite()
    
    # Save test suite if requested
    if args.save_suite:
        save_test_suite(test_suite, Path(args.suite_path))
    
    # Run evaluation
    try:
        report = run_evaluation(args.server_url, test_suite)
        print("\n🎉 Evaluation completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Evaluation failed: {str(e)}")


if __name__ == "__main__":
    main()