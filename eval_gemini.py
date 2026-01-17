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
import time
from pathlib import Path

from src.sledge_eval import GeminiEvaluator, VoiceCommandTest, ToolCall, EvaluationReport
from src.sledge_eval.gemini_evaluator import GeminiTextEvaluator
from src.sledge_eval.gemini_anki_evaluator import GeminiAnkiEvaluator


def generate_report(
    results: list,
    mode: str,
    model_name: str = "unknown",
    test_suite_name: str = None
) -> Path:
    """Generate and save a comprehensive evaluation report."""
    # Calculate total evaluation time
    total_time = sum(r.evaluation_time_ms or 0 for r in results)

    # Create report
    report = EvaluationReport(
        model_name=model_name,
        server_url="https://generativelanguage.googleapis.com",
        evaluation_mode=mode,
        test_suite_name=test_suite_name,
        hardware_info=None,  # Cloud API - no local hardware info
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


def run_single_test(evaluator: GeminiEvaluator, model_name: str):
    """Run a single test evaluation."""
    print("=" * 80)
    print("Single Test Evaluation (Gemini)")
    print("=" * 80)

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

    generate_report([result], "single", model_name)

    print("=" * 80)
    return result.passed


def run_test_suite(evaluator: GeminiEvaluator, test_file: str, model_name: str):
    """Run a full test suite evaluation."""
    print("=" * 80)
    print("Test Suite Evaluation (Gemini)")
    print("=" * 80)

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
    results = []

    for test in test_suite.tests:
        print(f"🧪 Test: {test.id}")
        print(f"   Command: '{test.voice_command}'")
        result = evaluator.evaluate_test(test)
        results.append(result)

        status = "PASS ✓" if result.passed else "FAIL ✗"
        time_str = f"{result.evaluation_time_ms:.1f}ms" if result.evaluation_time_ms else "N/A"
        print(f"   Result: {status} ({time_str})")

        if not result.passed and result.error:
            print(f"   Error: {result.error}")
        print()

    # Display results summary
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

    generate_report(results, "suite", model_name, test_suite.name)

    print("=" * 80)
    return passed_count == total_count


def run_all_tests(evaluator: GeminiEvaluator, test_file: str, model_name: str):
    """Run all available tests."""
    print("=" * 80)
    print("ALL TESTS EVALUATION (Gemini)")
    print("=" * 80)

    all_results = []
    total_start_time = time.time()

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

    custom_evaluator = GeminiEvaluator(
        api_key=evaluator.api_key,
        model=evaluator.model,
        available_tools=custom_tools,
        debug=evaluator.debug,
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
    anki_evaluator = GeminiAnkiEvaluator(
        api_key=evaluator.api_key,
        model=evaluator.model,
        debug=evaluator.debug,
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

    # 5. Run text evaluation tests
    print(f"\n📋 SECTION 5: Text Evaluation Tests")
    print("-" * 40)

    # Initialize text evaluator
    text_evaluator = GeminiTextEvaluator(
        api_key=evaluator.api_key,
        model=evaluator.model,
        debug=evaluator.debug,
    )

    # Load text evaluation test suite
    text_test_file = Path("tests/test_data/comprehensive_text_suite.json")

    if text_test_file.exists():
        try:
            text_test_suite = text_evaluator.load_text_test_suite(text_test_file)
            print(f"Loaded: {text_test_suite.name} ({len(text_test_suite.tests)} tests)")

            for test in text_test_suite.tests:
                print(f"🧪 Test: {test.id}")
                print(f"   Question: '{test.question}'")
                result = text_evaluator.evaluate_text_test(test)

                # Convert TextEvaluationResult to EvaluationResult for compatibility
                from src.sledge_eval.evaluator import EvaluationResult
                eval_result = EvaluationResult(
                    test_id=result.test_id,
                    passed=result.passed,
                    predicted_tool_calls=[],  # Empty for text evaluations
                    expected_tool_calls=[],   # Empty for text evaluations
                    error=result.error,
                    evaluation_time_ms=result.evaluation_time_ms,
                    voice_command=result.question,  # Use question as voice_command
                    test_description=result.test_description,
                    tags=result.tags
                )
                all_results.append(eval_result)

                status = "PASS ✓" if result.passed else "FAIL ✗"
                time_str = f"{result.evaluation_time_ms:.1f}ms" if result.evaluation_time_ms else "N/A"
                print(f"   Result: {status} ({time_str})")
                if result.passed:
                    print(f"   Answer: '{result.predicted_answer}'")
                else:
                    print(f"   Expected: '{result.expected_answer}'")
                    print(f"   Got: '{result.predicted_answer}'")

        except Exception as e:
            print(f"⚠️ Warning: Could not run text evaluation tests: {e}")
            print("   Continuing with other tests...")
    else:
        print("⚠️ Warning: Text evaluation test suite not found - skipping text evaluation tests")
        print(f"   Expected: {text_test_file}")

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
        print(f"   Average Test Time: {avg_time:.1f}ms")

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

    generate_report(all_results, "all", model_name)

    print("=" * 80)
    return passed_count == total_count


def run_custom_tools(evaluator: GeminiEvaluator, model_name: str):
    """Run evaluation with custom tool definitions."""
    print("=" * 80)
    print("Custom Tools Evaluation (Gemini)")
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
    custom_evaluator = GeminiEvaluator(
        api_key=evaluator.api_key,
        model=evaluator.model,
        available_tools=custom_tools,
        debug=evaluator.debug,
    )

    print(f"✅ Custom tools evaluator initialized!")

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
    result = custom_evaluator.evaluate_test(test)

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
    generate_report([result], "custom", model_name)

    print("=" * 80)
    return result.passed


def run_anki_tests(api_key: str, model_name: str, debug: bool = False):
    """Run evaluation with Anki's large tool set (13 tools)."""
    print("=" * 80)
    print("Anki Large Tool Set Evaluation (Gemini)")
    print("=" * 80)

    # Initialize evaluator with Anki tools
    print(f"\nInitializing Gemini Anki evaluator...")
    if debug:
        print("🐛 Debug mode enabled")
    evaluator = GeminiAnkiEvaluator(
        api_key=api_key,
        model=model_name,
        debug=debug,
    )

    print(f"📊 Tool Set Size: {evaluator.get_tool_count()} tools")
    print(f"🔧 Available Tools: {', '.join(evaluator.get_tool_names()[:5])}... (showing first 5)")
    print(f"✅ Gemini Anki evaluator initialized!")

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
    generate_report(results, "anki_large_toolset", model_name, test_suite.name)

    print("=" * 80)
    return passed_count == total_count


def run_text_evaluation(api_key: str, model_name: str, test_file: str = None, debug: bool = False):
    """Run text evaluation tests."""
    print("=" * 80)
    print("TEXT EVALUATION (Gemini)")
    print("=" * 80)

    # Initialize text evaluator
    print(f"\nInitializing Gemini text evaluator...")
    if debug:
        print("🐛 Debug mode enabled")

    text_evaluator = GeminiTextEvaluator(
        api_key=api_key,
        model=model_name,
        debug=debug,
    )

    print(f"✅ Gemini text evaluator initialized!")

    # Load text evaluation test suite
    if test_file is None:
        test_file = Path("tests/test_data/comprehensive_text_suite.json")
    else:
        test_file = Path(test_file)

    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False

    try:
        text_test_suite = text_evaluator.load_text_test_suite(test_file)
        print(f"\nLoaded: {text_test_suite.name} ({len(text_test_suite.tests)} tests)")
        if text_test_suite.description:
            print(f"Description: {text_test_suite.description}")

        results = []

        print("\n" + "-" * 60)
        print("RUNNING TEXT EVALUATION TESTS")
        print("-" * 60)

        for i, test in enumerate(text_test_suite.tests, 1):
            print(f"\n🧪 Test {i}/{len(text_test_suite.tests)}: {test.id}")
            print(f"   Question: '{test.question}'")
            print(f"   Expected: '{test.expected_answer}'")

            result = text_evaluator.evaluate_text_test(test)
            results.append(result)

            status = "PASS ✓" if result.passed else "FAIL ✗"
            time_str = f"{result.evaluation_time_ms:.1f}ms" if result.evaluation_time_ms else "N/A"
            print(f"   Result: {status} ({time_str})")
            print(f"   Answer: '{result.predicted_answer}'")

            if not result.passed:
                print(f"   ❌ Expected '{result.expected_answer}' but got '{result.predicted_answer}'")

            if result.error:
                print(f"   ⚠️  Error: {result.error}")

        # Summary
        passed_count = sum(1 for r in results if r.passed)
        total_count = len(results)
        pass_rate = (passed_count / total_count * 100) if total_count > 0 else 0
        total_time = sum(r.evaluation_time_ms or 0 for r in results)

        print("\n" + "=" * 80)
        print("TEXT EVALUATION RESULTS")
        print("=" * 80)

        print(f"\n📊 Statistics:")
        print(f"   Tests Passed: {passed_count}/{total_count} ({pass_rate:.1f}%)")
        print(f"   Total Time: {total_time:.1f}ms")

        if results:
            avg_time = total_time / len(results)
            print(f"   Average Test Time: {avg_time:.1f}ms")

        print(f"\n📋 Individual Results:")
        for result in results:
            status = "PASS ✓" if result.passed else "FAIL ✗"
            time_str = f"({result.evaluation_time_ms:.1f}ms)" if result.evaluation_time_ms else "(N/A)"
            print(f"   {status} {result.test_id} {time_str}")
            if not result.passed:
                print(f"      Expected: '{result.expected_answer}'")
                print(f"      Got: '{result.predicted_answer}'")

        # Generate report
        report = text_evaluator.create_text_evaluation_report(
            results=results,
            model_name=model_name,
            test_suite_name=text_test_suite.name,
            evaluation_mode="text_evaluation"
        )

        file_paths = report.save_to_file(Path("."))
        print(f"\n📄 Reports saved:")
        print(f"   • JSON: {file_paths['json']}")
        print(f"   • Markdown: {file_paths['markdown']}")

        print("=" * 80)
        return passed_count == total_count

    except Exception as e:
        print(f"❌ Text evaluation failed: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        return False


def main():
    """Main entry point for sledge-eval Gemini CLI."""
    parser = argparse.ArgumentParser(
        description="Sledge Eval - Voice command evaluation using Google Gemini API",
        prog="python eval_gemini.py"
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

    args = parser.parse_args()

    print(f"🚀 Sledge Eval Gemini Client")
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    if args.debug:
        print(f"Debug: ENABLED 🐛")
    print()

    try:
        success = False

        # For anki and text modes, we don't need to initialize the standard evaluator
        if args.mode == "anki":
            success = run_anki_tests(args.api_key, args.model, args.debug)
        elif args.mode == "text":
            success = run_text_evaluation(args.api_key, args.model, args.test_suite, args.debug)
        else:
            # Initialize the evaluator for other modes
            print("Initializing Gemini evaluator...")
            evaluator = GeminiEvaluator(
                api_key=args.api_key,
                model=args.model,
                debug=args.debug,
            )
            print(f"✅ Connected to Gemini API with model: {args.model}")

            if args.mode == "single":
                success = run_single_test(evaluator, args.model)
            elif args.mode == "suite":
                success = run_test_suite(evaluator, args.test_suite, args.model)
            elif args.mode == "custom":
                success = run_custom_tools(evaluator, args.model)
            elif args.mode == "all":
                success = run_all_tests(evaluator, args.test_suite, args.model)

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
        import traceback
        if args.debug:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
