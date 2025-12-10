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

from src.sledge_eval import ServerEvaluator, VoiceCommandTest, ToolCall, EvaluationReport, AnkiLargeToolSetEvaluator
from src.sledge_eval.hardware_detector import HardwareDetector


def generate_report(
    results: list, 
    server_url: str, 
    mode: str, 
    model_name: str = "unknown",
    test_suite_name: str = None
) -> Path:
    """Generate and save a comprehensive evaluation report."""
    # Calculate total evaluation time
    total_time = sum(r.evaluation_time_ms or 0 for r in results)
    
    # Detect hardware information
    hardware_detector = HardwareDetector()
    hardware_info = hardware_detector.extract_hardware_info()
    
    # Create report
    report = EvaluationReport(
        model_name=model_name,
        server_url=server_url,
        evaluation_mode=mode,
        test_suite_name=test_suite_name,
        hardware_info=hardware_info,
        total_tests=0,
        passed_tests=0,
        failed_tests=0,
        pass_rate=0.0,
        total_evaluation_time_ms=total_time,
        test_results=[]
    )
    
    # Add all results
    for result in results:
        report.add_result(result)
    
    # Save report
    base_path = Path.cwd()
    report_paths = report.save_to_file(base_path)
    
    print(f"\n📊 Reports saved:")
    print(f"   JSON: {report_paths['json']}")
    print(f"   Markdown: {report_paths['markdown']}")
    return report_paths


def extract_model_name_from_url(server_url: str, evaluator: ServerEvaluator = None) -> str:
    """Extract model name from server or use URL as fallback."""
    try:
        if evaluator:
            # Try to get model info from server
            import requests
            response = requests.get(f"{server_url}/v1/models", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if "data" in data and len(data["data"]) > 0:
                    return data["data"][0].get("id", "unknown")
                elif "models" in data and len(data["models"]) > 0:
                    return data["models"][0].get("model", "unknown")
    except Exception:
        pass
    
    # Fallback to URL-based name
    return f"server_{server_url.replace('http://', '').replace('https://', '').replace(':', '_').replace('/', '_')}"


def run_single_test(server_url: str, debug: bool = False, model_name: str = None):
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
    
    # Display hardware information
    hardware_detector = HardwareDetector()
    hardware_summary = hardware_detector.get_hardware_summary()
    if hardware_summary:
        print(f"\n🖥️ Hardware Information:")
        for key, value in hardware_summary.items():
            print(f"   {key}: {value}")

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

    # Generate report
    if model_name is None:
        model_name = extract_model_name_from_url(server_url, evaluator)
    
    generate_report([result], server_url, "single", model_name)

    print("=" * 80)
    return result.passed


def run_anki_large_toolset(server_url: str, debug: bool = False, model_name: str = None):
    """Run evaluation with Anki's large tool set (26+ tools)."""
    print("=" * 80)
    print("Anki Large Tool Set Evaluation (Server)")
    print("=" * 80)

    # Initialize evaluator with Anki tools
    print(f"\nConnecting to llama-server at: {server_url}")
    if debug:
        print("🐛 Debug mode enabled")
    evaluator = AnkiLargeToolSetEvaluator(
        server_url=server_url,
        debug=debug,
    )
    
    print(f"📊 Tool Set Size: {evaluator.get_tool_count()} tools")
    print(f"🔧 Available Tools: {', '.join(evaluator.get_tool_names()[:5])}... (showing first 5)")
    
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
    
    print("✅ Server is responding with large tool set!")

    # Load Anki test suite
    test_file = Path("tests/test_data/anki_large_toolset_suite.json")
    
    if not test_file.exists():
        print(f"❌ Test suite file not found: {test_file}")
        print("   Make sure you have the Anki test suite file in the correct location.")
        return False

    print(f"\nLoading Anki large toolset test suite from: {test_file}")
    test_suite = evaluator.load_test_suite(Path(test_file))
    print(f"Loaded test suite: {test_suite.name}")
    print(f"Description: {test_suite.description}")
    print(f"Number of tests: {len(test_suite.tests)}\n")

    # Evaluate the suite
    print("Running large toolset evaluation...\n")
    results = []
    total_start_time = time.time()
    
    for test in test_suite.tests:
        print(f"🧪 Test: {test.id}")
        print(f"   Command: '{test.voice_command}'")
        print(f"   Tags: {', '.join(test.tags)}")
        result = evaluator.evaluate_test(test)
        results.append(result)
        
        status = "PASS ✓" if result.passed else "FAIL ✗"
        time_str = f"{result.evaluation_time_ms:.1f}ms" if result.evaluation_time_ms else "N/A"
        print(f"   Result: {status} ({time_str})")
        
        if not result.passed and result.error:
            print(f"   Error: {result.error}")
        print()

    # Calculate summary metrics
    total_time = (time.time() - total_start_time) * 1000
    passed_count = sum(1 for r in results if r.passed)
    total_count = len(results)
    pass_rate = (passed_count / total_count * 100) if total_count > 0 else 0

    print("\n" + "=" * 80)
    print("ANKI LARGE TOOLSET RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n📊 Overall Statistics:")
    print(f"   Tools Available: {evaluator.get_tool_count()}")
    print(f"   Tests Passed: {passed_count}/{total_count} ({pass_rate:.1f}%)")
    print(f"   Total Time: {total_time:.1f}ms")
    
    if results:
        avg_time = sum(r.evaluation_time_ms for r in results if r.evaluation_time_ms) / len([r for r in results if r.evaluation_time_ms])
        print(f"   Average Test Time: {avg_time:.1f}ms")

    # Analyze performance by complexity
    complexity_stats = {}
    for result in results:
        for tag in result.tags:
            if tag in ['basic', 'intermediate', 'advanced', 'expert']:
                if tag not in complexity_stats:
                    complexity_stats[tag] = {'passed': 0, 'total': 0}
                complexity_stats[tag]['total'] += 1
                if result.passed:
                    complexity_stats[tag]['passed'] += 1

    if complexity_stats:
        print(f"\n📈 Performance by Complexity:")
        for complexity, stats in sorted(complexity_stats.items()):
            rate = (stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"   {complexity.title()}: {stats['passed']}/{stats['total']} ({rate:.1f}%)")

    print(f"\n📋 Individual Results:")
    for result in results:
        status = "PASS ✓" if result.passed else "FAIL ✗" 
        time_str = f"({result.evaluation_time_ms:.1f}ms)" if result.evaluation_time_ms else "(N/A)"
        complexity = next((tag for tag in result.tags if tag in ['basic', 'intermediate', 'advanced', 'expert']), 'other')
        print(f"   {status} {result.test_id} [{complexity}] {time_str}")

        if not result.passed:
            print(f"      Expected: {[f'{tc.name}({tc.arguments})' for tc in result.expected_tool_calls]}")
            print(f"      Predicted: {[f'{tc.name}({tc.arguments})' for tc in result.predicted_tool_calls]}")
            if result.error:
                print(f"      Error: {result.error}")

    # Generate report
    if model_name is None:
        model_name = extract_model_name_from_url(server_url, evaluator)
    
    generate_report(results, server_url, "anki_large_toolset", model_name, test_suite.name)

    print("=" * 80)
    return passed_count == total_count


def run_all_tests(server_url: str, test_file: str = None, debug: bool = False, model_name: str = None):
    """Run all test types: single test, test suite, custom tools, and Anki large toolset."""
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
    
    # Display hardware information
    hardware_detector = HardwareDetector()
    hardware_summary = hardware_detector.get_hardware_summary()
    if hardware_summary:
        print(f"\n🖥️ Hardware Information:")
        for key, value in hardware_summary.items():
            print(f"   {key}: {value}")
    
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
    
    # 4. Run Anki large toolset test
    print(f"\n📋 SECTION 4: Anki Large Toolset Test")
    print("-" * 40)
    
    # Initialize Anki evaluator
    anki_evaluator = AnkiLargeToolSetEvaluator(
        server_url=server_url,
        debug=debug,
    )
    
    print(f"🔧 Tool Set Size: {anki_evaluator.get_tool_count()} tools")
    
    # Load Anki test suite
    anki_test_file = Path("tests/test_data/anki_large_toolset_suite.json")
    
    if anki_test_file.exists():
        try:
            anki_test_suite = anki_evaluator.load_test_suite(anki_test_file)
            print(f"Loaded: {anki_test_suite.name} ({len(anki_test_suite.tests)} tests)")
            
            for test in anki_test_suite.tests:
                print(f"🧪 Test: {test.id}")
                print(f"   Command: '{test.voice_command}'")
                result = anki_evaluator.evaluate_test(test)
                all_results.append(result)
                
                status = "PASS ✓" if result.passed else "FAIL ✗"
                time_str = f"{result.evaluation_time_ms:.1f}ms" if result.evaluation_time_ms else "N/A"
                print(f"   Result: {status} ({time_str})")
                
        except Exception as e:
            print(f"⚠️ Warning: Could not run Anki tests: {e}")
            print("   Continuing with other tests...")
    else:
        print("⚠️ Warning: Anki test suite not found - skipping large toolset tests")
        print(f"   Expected: {anki_test_file}")

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

    # Generate report
    if model_name is None:
        model_name = extract_model_name_from_url(server_url, evaluator)
    
    generate_report(all_results, server_url, "all", model_name)

    print("=" * 80)
    return passed_count == total_count


def run_test_suite(server_url: str, test_file: str = None, debug: bool = False, model_name: str = None):
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

    # Generate report
    if model_name is None:
        model_name = extract_model_name_from_url(server_url, evaluator)
    
    generate_report(results, server_url, "suite", model_name, test_suite.name)

    print("=" * 80)
    return passed_count == total_count


def run_custom_tools(server_url: str, debug: bool = False, model_name: str = None):
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
    
    # Generate report
    if model_name is None:
        model_name = extract_model_name_from_url(server_url, evaluator)
    
    generate_report([result], server_url, "custom", model_name)
    
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
        choices=["single", "suite", "custom", "all", "anki"],
        default="all",
        help="Evaluation mode: single test, test suite, custom tools, all tests combined, or anki large toolset (default: all)",
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
    parser.add_argument(
        "--model-name",
        type=str,
        help="Override model name for report generation (will auto-detect from server if not provided)",
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
            success = run_single_test(server_url, args.debug, args.model_name)
        elif args.mode == "suite":
            success = run_test_suite(server_url, args.test_suite, args.debug, args.model_name)
        elif args.mode == "custom":
            success = run_custom_tools(server_url, args.debug, args.model_name)
        elif args.mode == "anki":
            success = run_anki_large_toolset(server_url, args.debug, args.model_name)
        elif args.mode == "all":
            success = run_all_tests(server_url, args.test_suite, args.debug, args.model_name)

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