#!/usr/bin/env python3
"""
Sledge Eval CLI for OpenRouter API

Entry point for running voice command evaluations against models hosted on OpenRouter.

Usage:
    python eval_openrouter.py --model anthropic/claude-3-haiku
    python eval_openrouter.py --model openai/gpt-4o --mode all --debug
    python eval_openrouter.py --model meta-llama/llama-3-70b-instruct --api-key sk-or-xxx
"""

import argparse
from pathlib import Path

from src.sledge_eval.cli import OpenRouterRunner


def main():
    """Main entry point for sledge-eval OpenRouter CLI."""
    parser = argparse.ArgumentParser(
        description="Sledge Eval - Voice command evaluation using OpenRouter API",
        prog="python eval_openrouter.py",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="OpenRouter model ID (e.g., 'anthropic/claude-3-haiku', 'openai/gpt-4o')",
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
        help="OpenRouter API key (defaults to OPENROUTER_API_KEY env var or .env file)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--site-url",
        type=str,
        help="Optional site URL for OpenRouter ranking (HTTP-Referer header)",
    )
    parser.add_argument(
        "--app-name",
        type=str,
        default="sledge-eval",
        help="App name for OpenRouter ranking (X-Title header, default: sledge-eval)",
    )

    args = parser.parse_args()

    print(f"Sledge Eval OpenRouter Client")
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    if args.debug:
        print(f"Debug: ENABLED")
    print()

    try:
        # Create runner
        runner = OpenRouterRunner(
            model=args.model,
            api_key=args.api_key,
            timeout=args.timeout,
            debug=args.debug,
            site_url=args.site_url,
            app_name=args.app_name,
        )

        # Parse test file path if provided
        test_file = Path(args.test_suite) if args.test_suite else None

        # Run the appropriate mode
        success = False
        if args.mode == "single":
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
            print("\nAll tests passed!")
            return 0
        else:
            print("\nSome tests failed!")
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
