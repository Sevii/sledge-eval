#!/usr/bin/env python3
"""
Sledge Eval CLI for llama-server

Entry point for running voice command evaluations against llama-server.

Usage:
    python eval_server.py --port 8080
    python eval_server.py --server-url http://localhost:8080 --mode suite
    python eval_server.py --port 8080 --mode custom
"""

import argparse
from pathlib import Path

from src.sledge_eval import ServerEvaluator, VoiceCommandTest, ToolCall


def run_single_test(server_url: str, debug: bool = False):
    """Run a single test evaluation."""
    print("=" * 80)
    print("Single Test Evaluation (Server)")
    print("=" * 80)

    # Initialize the evaluator
    print(f"\nConnecting to llama-server at: {server_url}")
    if debug:
        print("🐛 Debug mode enabled")
    evaluator = ServerEvaluator(server_url=server_url, debug=debug)
    
    # Check server health
    if not evaluator._check_server_health():
        print("❌ Server is not responding. Make sure llama-server is running.")
        print(f"   Base URL: {server_url}")
        
        # Check if anything is listening on the port
        import subprocess
        try:
            port = server_url.split(':')[-1]
            result = subprocess.run(['lsof', '-i', f':{port}'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"   Something is listening on port {port}:")
                print("   " + result.stdout.replace('\n', '\n   '))
            else:
                print(f"   Nothing is listening on port {port}")
        except Exception:
            pass
            
        print("   Possible issues:")
        print("   - Server not started yet")
        print("   - Wrong port number")  
        print("   - Server crashed during startup")
        print("   - Firewall blocking connection")
        print("   - Server doesn't support expected API endpoints")
        return False
    
    print("✅ Server is responding!")

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

    print(f"\nTest ID: {test.id}")
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
    return result.passed


def run_test_suite(server_url: str, test_file: str = None, debug: bool = False):
    """Run a full test suite evaluation."""
    print("=" * 80)
    print("Test Suite Evaluation (Server)")
    print("=" * 80)

    # Initialize the evaluator
    print(f"\nConnecting to llama-server at: {server_url}")
    if debug:
        print("🐛 Debug mode enabled")
    evaluator = ServerEvaluator(server_url=server_url, debug=debug)
    
    # Check server health
    if not evaluator._check_server_health():
        print("❌ Server is not responding. Make sure llama-server is running.")
        print(f"   Base URL: {server_url}")
        
        # Check if anything is listening on the port
        import subprocess
        try:
            port = server_url.split(':')[-1]
            result = subprocess.run(['lsof', '-i', f':{port}'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"   Something is listening on port {port}:")
                print("   " + result.stdout.replace('\n', '\n   '))
            else:
                print(f"   Nothing is listening on port {port}")
        except Exception:
            pass
            
        print("   Possible issues:")
        print("   - Server not started yet")
        print("   - Wrong port number")  
        print("   - Server crashed during startup")
        print("   - Firewall blocking connection")
        print("   - Server doesn't support expected API endpoints")
        return False
    
    print("✅ Server is responding!")

    # Load test suite
    if test_file is None:
        test_file = Path("tests/test_data/example_test_suite.json")

    print(f"\nLoading test suite from: {test_file}")
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
    return passed_count == total_count


def run_custom_tools(server_url: str, debug: bool = False):
    """Run evaluation with custom tool definitions."""
    print("=" * 80)
    print("Custom Tools Evaluation (Server)")
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
    print(f"\nConnecting to llama-server at: {server_url}")
    if debug:
        print("🐛 Debug mode enabled")
    evaluator = ServerEvaluator(
        server_url=server_url,
        available_tools=custom_tools,
        debug=debug,
    )
    
    # Check server health
    if not evaluator._check_server_health():
        print("❌ Server is not responding. Make sure llama-server is running.")
        print(f"   Base URL: {server_url}")
        
        # Check if anything is listening on the port
        import subprocess
        try:
            port = server_url.split(':')[-1]
            result = subprocess.run(['lsof', '-i', f':{port}'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"   Something is listening on port {port}:")
                print("   " + result.stdout.replace('\n', '\n   '))
            else:
                print(f"   Nothing is listening on port {port}")
        except Exception:
            pass
            
        print("   Possible issues:")
        print("   - Server not started yet")
        print("   - Wrong port number")  
        print("   - Server crashed during startup")
        print("   - Firewall blocking connection")
        print("   - Server doesn't support expected API endpoints")
        return False
    
    print("✅ Server is responding with custom tools!")

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

    print(f"\nVoice Command: '{test.voice_command}'")
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
    return result.passed


def main():
    """Main entry point for sledge-eval server CLI."""
    parser = argparse.ArgumentParser(
        description="Sledge Eval - Voice command evaluation using llama-server",
        prog="python eval_server.py"
    )
    parser.add_argument(
        "--server-url",
        type=str,
        help="URL of the llama-server instance (default: http://localhost:PORT)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port of the llama-server instance (default: 8080)",
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
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging of requests and responses",
    )

    args = parser.parse_args()

    # Determine server URL
    if args.server_url:
        server_url = args.server_url
    else:
        server_url = f"http://localhost:{args.port}"

    print(f"🚀 Sledge Eval Server Client")
    print(f"Server URL: {server_url}")
    print(f"Mode: {args.mode}")
    print(f"Timeout: {args.timeout}s")
    if args.debug:
        print(f"Debug: ENABLED 🐛")
    print()

    try:
        success = False
        if args.mode == "single":
            success = run_single_test(server_url, args.debug)
        elif args.mode == "suite":
            success = run_test_suite(server_url, args.test_suite, args.debug)
        elif args.mode == "custom":
            success = run_custom_tools(server_url, args.debug)

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
        return 1


if __name__ == "__main__":
    exit(main())