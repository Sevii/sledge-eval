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
import time
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
    
    if result.evaluation_time_ms:
        print(f"Evaluation Time: {result.evaluation_time_ms:.1f}ms")
    
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


def run_all_tests(server_url: str, test_file: str = None, debug: bool = False):
    """Run all test types: single test, test suite, and custom tools."""
    print("=" * 80)
    print("ALL TESTS EVALUATION (Server)")
    print("=" * 80)
    
    all_results = []
    total_start_time = time.time()
    
    # Initialize the evaluator
    print(f"\nConnecting to llama-server at: {server_url}")
    if debug:
        print("🐛 Debug mode enabled")
    
    # Check server health once
    evaluator = ServerEvaluator(server_url=server_url, debug=debug)
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
    
    print("\n" + "=" * 80)
    print("RUNNING ALL TESTS")
    print("=" * 80)
    
    # 1. Run single test
    print(f"\n📋 SECTION 1: Single Test")
    print("-" * 40)
    
    single_test = VoiceCommandTest(
        id="single_001",
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
    
    print(f"🧪 Test: {single_test.id}")
    print(f"   Command: '{single_test.voice_command}'")
    result = evaluator.evaluate_test(single_test)
    all_results.append(result)
    
    status = "PASS ✓" if result.passed else "FAIL ✗"
    time_str = f"{result.evaluation_time_ms:.1f}ms" if result.evaluation_time_ms else "N/A"
    print(f"   Result: {status} ({time_str})")
    
    # 2. Run test suite
    print(f"\n📋 SECTION 2: Test Suite")
    print("-" * 40)
    
    if test_file is None:
        test_file = Path("tests/test_data/example_test_suite.json")

    test_suite = evaluator.load_test_suite(Path(test_file))
    print(f"Loaded: {test_suite.name} ({len(test_suite.tests)} tests)")
    
    for test in test_suite.tests:
        print(f"🧪 Test: {test.id}")
        print(f"   Command: '{test.voice_command}'")
        result = evaluator.evaluate_test(test)
        all_results.append(result)
        
        status = "PASS ✓" if result.passed else "FAIL ✗"
        time_str = f"{result.evaluation_time_ms:.1f}ms" if result.evaluation_time_ms else "N/A"
        print(f"   Result: {status} ({time_str})")
    
    # 3. Run custom tools test
    print(f"\n📋 SECTION 3: Custom Tools Test")
    print("-" * 40)
    
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

    custom_evaluator = ServerEvaluator(
        server_url=server_url,
        available_tools=custom_tools,
        debug=debug,
    )

    custom_test = VoiceCommandTest(
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

    print(f"🧪 Test: {custom_test.id}")
    print(f"   Command: '{custom_test.voice_command}'")
    result = custom_evaluator.evaluate_test(custom_test)
    all_results.append(result)
    
    status = "PASS ✓" if result.passed else "FAIL ✗"
    time_str = f"{result.evaluation_time_ms:.1f}ms" if result.evaluation_time_ms else "N/A"
    print(f"   Result: {status} ({time_str})")

    # Summary
    total_time = (time.time() - total_start_time) * 1000
    passed_count = sum(1 for r in all_results if r.passed)
    total_count = len(all_results)
    pass_rate = (passed_count / total_count * 100) if total_count > 0 else 0

    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n📊 Overall Statistics:")
    print(f"   Tests Passed: {passed_count}/{total_count} ({pass_rate:.1f}%)")
    print(f"   Total Time: {total_time:.1f}ms")
    
    if all_results:
        avg_time = sum(r.evaluation_time_ms for r in all_results if r.evaluation_time_ms) / len([r for r in all_results if r.evaluation_time_ms])
        min_time = min(r.evaluation_time_ms for r in all_results if r.evaluation_time_ms)
        max_time = max(r.evaluation_time_ms for r in all_results if r.evaluation_time_ms)
        print(f"   Average Test Time: {avg_time:.1f}ms")
        print(f"   Fastest Test: {min_time:.1f}ms")
        print(f"   Slowest Test: {max_time:.1f}ms")

    print(f"\n📋 Individual Results:")
    for result in all_results:
        status = "PASS ✓" if result.passed else "FAIL ✗"
        time_str = f"({result.evaluation_time_ms:.1f}ms)" if result.evaluation_time_ms else "(N/A)"
        print(f"   {status} {result.test_id} {time_str}")

        if not result.passed:
            print(f"      Expected: {[f'{tc.name}({tc.arguments})' for tc in result.expected_tool_calls]}")
            print(f"      Predicted: {[f'{tc.name}({tc.arguments})' for tc in result.predicted_tool_calls]}")
            if result.error:
                print(f"      Error: {result.error}")

    print("=" * 80)
    return passed_count == total_count


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
        time_str = f"({result.evaluation_time_ms:.1f}ms)" if result.evaluation_time_ms else ""
        print(f"{status} {result.test_id} {time_str}")

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
    
    if result.evaluation_time_ms:
        print(f"Evaluation Time: {result.evaluation_time_ms:.1f}ms")
    
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
        choices=["single", "suite", "custom", "all"],
        default="all",
        help="Evaluation mode: single test, test suite, custom tools, or all tests combined (default: all)",
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
        elif args.mode == "all":
            success = run_all_tests(server_url, args.test_suite, args.debug)

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