#!/usr/bin/env python3
"""
Sledge Eval CLI for Google Gemini API

Entry point for running voice command evaluations against Gemini models.

Usage:
    python eval_gemini.py
    python eval_gemini.py --model gemini-2.5-flash-lite
    python eval_gemini.py --mode suite --test-suite tests/test_data/example_test_suite.json
    python eval_gemini.py --debug
"""

import argparse
from pathlib import Path

from src.sledge_eval.cli import GeminiRunner


def main():
    """Main entry point for sledge-eval Gemini CLI."""
    parser = argparse.ArgumentParser(
        description="Sledge Eval - Voice command evaluation using Google Gemini API",
        prog="python eval_gemini.py",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash-lite",
        help="Gemini model to use (default: gemini-2.5-flash-lite)",
    )
    parser.add_argument(
        "--test-suite",
        type=str,
        help="Path to test suite JSON file (optional)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "suite", "custom", "all", "anki", "text"],
        default="all",
        help="Evaluation mode: single test, test suite, custom tools, all tests combined, anki large toolset, or text evaluation (default: all)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging of requests and responses",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="Google API key (defaults to GEMINI_API_KEY env var or APIKey from .env)",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        help="Path to prompts JSON file for prompt comparison mode",
    )

    args = parser.parse_args()

    print(f"🚀 Sledge Eval Gemini Client")
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    if args.debug:
        print(f"Debug: ENABLED 🐛")
    print()

    try:
        # Create runner
        runner = GeminiRunner(
            model=args.model,
            api_key=args.api_key,
            debug=args.debug,
        )

        # Parse test file path if provided
        test_file = Path(args.test_suite) if args.test_suite else None
        prompts_file = Path(args.prompts_file) if args.prompts_file else None

        # Run prompt comparison if prompts file is provided
        if prompts_file:
            success = runner.run_prompt_comparison(prompts_file, test_file)
        # Otherwise run the appropriate mode
        elif args.mode == "single":
            success = runner.run_single_test()
        elif args.mode == "suite":
            success = runner.run_test_suite(test_file)
        elif args.mode == "custom":
            success = runner.run_custom_tools()
        elif args.mode == "anki":
            success = runner.run_anki_tests()
        elif args.mode == "text":
            success = runner.run_text_evaluation(test_file)
        elif args.mode == "all":
            success = runner.run_all_tests(test_file)

        if success:
            print("\n🎉 All tests passed!")
            return 0
        else:
            print("\n❌ Some tests failed!")
            return 1

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
