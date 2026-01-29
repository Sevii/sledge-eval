"""Shared evaluation runner for CLI tools."""

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from ..evaluator import (
    EvaluationResult,
    Evaluator,
    TestSuite,
    ToolCall,
    VoiceCommandTest,
    TextEvaluationSuite,
)
from .report_generator import ReportGenerator


# Default test file paths
DEFAULT_TEST_SUITE = Path("tests/test_data/example_test_suite.json")
DEFAULT_ANKI_TEST_SUITE = Path("tests/test_data/anki_large_toolset_suite.json")
DEFAULT_TEXT_TEST_SUITE = Path("tests/test_data/comprehensive_text_suite.json")


class EvaluationRunner(ABC):
    """Base class for running evaluations with different backends."""

    def __init__(
        self,
        model_name: str,
        debug: bool = False,
        report_generator: Optional[ReportGenerator] = None,
    ):
        """
        Initialize the evaluation runner.

        Args:
            model_name: Name of the model being evaluated
            debug: Enable debug logging
            report_generator: Optional custom report generator
        """
        self.model_name = model_name
        self.debug = debug
        self.report_generator = report_generator or ReportGenerator()

    @abstractmethod
    def get_evaluator(self) -> Evaluator:
        """Get the primary evaluator instance."""
        pass

    @abstractmethod
    def get_server_url(self) -> Optional[str]:
        """Get the server URL if applicable."""
        pass

    @abstractmethod
    def check_connection(self) -> bool:
        """Check if the evaluation backend is available."""
        pass

    def get_hosting_provider(self) -> Optional[str]:
        """Get the hosting provider name. Override in subclasses."""
        return None

    def get_custom_evaluator(self, custom_tools: List[Dict[str, Any]]) -> Evaluator:
        """
        Get an evaluator with custom tools.

        Override this method for backends that support custom tools.
        """
        raise NotImplementedError("Custom tools not supported for this backend")

    def get_anki_evaluator(self) -> Optional[Evaluator]:
        """
        Get an Anki evaluator if supported.

        Override this method for backends that support Anki evaluations.
        """
        return None

    def get_text_evaluator(self) -> Optional[Any]:
        """
        Get a text evaluator if supported.

        Override this method for backends that support text evaluations.
        """
        return None

    def run_single_test(self) -> bool:
        """Run a single test evaluation."""
        self._print_header("Single Test Evaluation")

        if not self._verify_connection():
            return False

        evaluator = self.get_evaluator()

        # Create a standard test case
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

        # Evaluate
        print("Evaluating...")
        result = evaluator.evaluate_test(test)

        # Display results
        self._print_single_result(result)

        # Generate report
        report_paths = self.report_generator.generate_report(
            results=[result],
            model_name=self.model_name,
            mode="single",
            server_url=self.get_server_url(),
            hosting_provider=self.get_hosting_provider(),
        )
        self.report_generator.print_report_paths(report_paths)

        print("=" * 80)
        return result.passed

    def run_test_suite(self, test_file: Optional[Path] = None) -> bool:
        """Run a full test suite evaluation."""
        self._print_header("Test Suite Evaluation")

        if not self._verify_connection():
            return False

        evaluator = self.get_evaluator()
        test_file = test_file or DEFAULT_TEST_SUITE

        print(f"\nLoading test suite from: {test_file}")
        test_suite = evaluator.load_test_suite(test_file)
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

        # Display summary
        self._print_suite_summary(results)

        # Generate report
        report_paths = self.report_generator.generate_report(
            results=results,
            model_name=self.model_name,
            mode="suite",
            server_url=self.get_server_url(),
            hosting_provider=self.get_hosting_provider(),
            test_suite_name=test_suite.name,
        )
        self.report_generator.print_report_paths(report_paths)

        print("=" * 80)
        passed_count = sum(1 for r in results if r.passed)
        return passed_count == len(results)

    def run_custom_tools(self) -> bool:
        """Run evaluation with custom tool definitions."""
        self._print_header("Custom Tools Evaluation")

        if not self._verify_connection():
            return False

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

        evaluator = self.get_custom_evaluator(custom_tools)
        print("✅ Custom tools evaluator initialized!")

        # Create a test
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
        self._print_single_result(result)

        # Generate report
        report_paths = self.report_generator.generate_report(
            results=[result],
            model_name=self.model_name,
            mode="custom",
            server_url=self.get_server_url(),
            hosting_provider=self.get_hosting_provider(),
        )
        self.report_generator.print_report_paths(report_paths)

        print("=" * 80)
        return result.passed

    def run_anki_tests(self) -> bool:
        """Run evaluation with Anki's large tool set."""
        self._print_header("Anki Large Tool Set Evaluation")

        anki_evaluator = self.get_anki_evaluator()
        if anki_evaluator is None:
            print("❌ Anki evaluator not available for this backend")
            return False

        if not self._verify_connection():
            return False

        print(f"📊 Tool Set Size: {anki_evaluator.get_tool_count()} tools")
        print(f"🔧 Available Tools: {', '.join(anki_evaluator.get_tool_names()[:5])}... (showing first 5)")

        test_file = DEFAULT_ANKI_TEST_SUITE
        if not test_file.exists():
            print(f"❌ Test suite file not found: {test_file}")
            return False

        print(f"\nLoading Anki large toolset test suite from: {test_file}")
        test_suite = anki_evaluator.load_test_suite(test_file)
        print(f"Loaded test suite: {test_suite.name}")
        print(f"Description: {test_suite.description}")
        print(f"Number of tests: {len(test_suite.tests)}\n")

        # Evaluate
        print("Running large toolset evaluation...\n")
        results = []
        total_start_time = time.time()

        for test in test_suite.tests:
            print(f"🧪 Test: {test.id}")
            print(f"   Command: '{test.voice_command}'")
            print(f"   Tags: {', '.join(test.tags)}")
            result = anki_evaluator.evaluate_test(test)
            results.append(result)

            status = "PASS ✓" if result.passed else "FAIL ✗"
            time_str = f"{result.evaluation_time_ms:.1f}ms" if result.evaluation_time_ms else "N/A"
            print(f"   Result: {status} ({time_str})")

            if not result.passed and result.error:
                print(f"   Error: {result.error}")
            print()

        # Summary
        self._print_anki_summary(results, anki_evaluator.get_tool_count(), total_start_time)

        # Generate report
        report_paths = self.report_generator.generate_report(
            results=results,
            model_name=self.model_name,
            mode="anki_large_toolset",
            server_url=self.get_server_url(),
            hosting_provider=self.get_hosting_provider(),
            test_suite_name=test_suite.name,
        )
        self.report_generator.print_report_paths(report_paths)

        print("=" * 80)
        passed_count = sum(1 for r in results if r.passed)
        return passed_count == len(results)

    def run_text_evaluation(self, test_file: Optional[Path] = None) -> bool:
        """Run text evaluation tests."""
        self._print_header("Text Evaluation")

        text_evaluator = self.get_text_evaluator()
        if text_evaluator is None:
            print("❌ Text evaluator not available for this backend")
            return False

        if not self._verify_connection():
            return False

        print("✅ Text evaluator initialized!")

        test_file = test_file or DEFAULT_TEXT_TEST_SUITE
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
            self._print_text_summary(results)

            # Generate report
            report = text_evaluator.create_text_evaluation_report(
                results=results,
                model_name=self.model_name,
                test_suite_name=text_test_suite.name,
                evaluation_mode="text_evaluation",
            )

            file_paths = report.save_to_file(Path("."))
            print(f"\n📄 Reports saved:")
            print(f"   • JSON: {file_paths['json']}")
            print(f"   • Markdown: {file_paths['markdown']}")

            print("=" * 80)
            passed_count = sum(1 for r in results if r.passed)
            return passed_count == len(results)

        except Exception as e:
            print(f"❌ Text evaluation failed: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return False

    def run_all_tests(self, test_file: Optional[Path] = None) -> bool:
        """Run all available tests."""
        self._print_header("ALL TESTS EVALUATION")

        if not self._verify_connection():
            return False

        all_results = []
        total_start_time = time.time()

        evaluator = self.get_evaluator()

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
        self._print_test_result(result)

        # 2. Run test suite
        print(f"\n📋 SECTION 2: Test Suite")
        print("-" * 40)

        test_file = test_file or DEFAULT_TEST_SUITE
        test_suite = evaluator.load_test_suite(test_file)
        print(f"Loaded: {test_suite.name} ({len(test_suite.tests)} tests)")

        for test in test_suite.tests:
            print(f"🧪 Test: {test.id}")
            print(f"   Command: '{test.voice_command}'")
            result = evaluator.evaluate_test(test)
            all_results.append(result)
            self._print_test_result(result)

        # 3. Run custom tools test
        print(f"\n📋 SECTION 3: Custom Tools Test")
        print("-" * 40)

        try:
            custom_tools = self._get_custom_email_tools()
            custom_evaluator = self.get_custom_evaluator(custom_tools)

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
            self._print_test_result(result)
        except NotImplementedError:
            print("⚠️ Custom tools not supported - skipping")

        # 4. Run Anki tests
        print(f"\n📋 SECTION 4: Anki Large Toolset Test")
        print("-" * 40)

        anki_evaluator = self.get_anki_evaluator()
        if anki_evaluator:
            print(f"🔧 Tool Set Size: {anki_evaluator.get_tool_count()} tools")

            anki_test_file = DEFAULT_ANKI_TEST_SUITE
            if anki_test_file.exists():
                try:
                    anki_test_suite = anki_evaluator.load_test_suite(anki_test_file)
                    print(f"Loaded: {anki_test_suite.name} ({len(anki_test_suite.tests)} tests)")

                    for test in anki_test_suite.tests:
                        print(f"🧪 Test: {test.id}")
                        print(f"   Command: '{test.voice_command}'")
                        result = anki_evaluator.evaluate_test(test)
                        all_results.append(result)
                        self._print_test_result(result)
                except Exception as e:
                    print(f"⚠️ Warning: Could not run Anki tests: {e}")
            else:
                print(f"⚠️ Warning: Anki test suite not found at {anki_test_file}")
        else:
            print("⚠️ Anki evaluator not available - skipping")

        # 5. Run text evaluation tests
        print(f"\n📋 SECTION 5: Text Evaluation Tests")
        print("-" * 40)

        text_evaluator = self.get_text_evaluator()
        if text_evaluator:
            text_test_file = DEFAULT_TEXT_TEST_SUITE
            if text_test_file.exists():
                try:
                    text_test_suite = text_evaluator.load_text_test_suite(text_test_file)
                    print(f"Loaded: {text_test_suite.name} ({len(text_test_suite.tests)} tests)")

                    for test in text_test_suite.tests:
                        print(f"🧪 Test: {test.id}")
                        print(f"   Question: '{test.question}'")
                        result = text_evaluator.evaluate_text_test(test)

                        # Convert TextEvaluationResult to EvaluationResult
                        from ..evaluator import EvaluationResult
                        eval_result = EvaluationResult(
                            test_id=result.test_id,
                            passed=result.passed,
                            predicted_tool_calls=[],
                            expected_tool_calls=[],
                            error=result.error,
                            evaluation_time_ms=result.evaluation_time_ms,
                            voice_command=result.question,
                            test_description=result.test_description,
                            tags=result.tags,
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
            else:
                print(f"⚠️ Warning: Text test suite not found at {text_test_file}")
        else:
            print("⚠️ Text evaluator not available - skipping")

        # Final summary
        self._print_final_summary(all_results, total_start_time)

        # Generate report
        report_paths = self.report_generator.generate_report(
            results=all_results,
            model_name=self.model_name,
            mode="all",
            server_url=self.get_server_url(),
            hosting_provider=self.get_hosting_provider(),
        )
        self.report_generator.print_report_paths(report_paths)

        print("=" * 80)
        passed_count = sum(1 for r in all_results if r.passed)
        return passed_count == len(all_results)

    # Helper methods

    def _print_header(self, title: str) -> None:
        """Print a section header."""
        print("=" * 80)
        print(title)
        print("=" * 80)

    def _verify_connection(self) -> bool:
        """Verify connection to the evaluation backend."""
        if not self.check_connection():
            print("❌ Connection check failed.")
            return False
        print("✅ Connection verified!")
        return True

    def _print_test_result(self, result: EvaluationResult) -> None:
        """Print a single test result."""
        status = "PASS ✓" if result.passed else "FAIL ✗"
        time_str = f"{result.evaluation_time_ms:.1f}ms" if result.evaluation_time_ms else "N/A"
        print(f"   Result: {status} ({time_str})")

    def _print_single_result(self, result: EvaluationResult) -> None:
        """Print detailed result for a single test."""
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

    def _print_suite_summary(self, results: List[EvaluationResult]) -> None:
        """Print summary for a test suite."""
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)

        passed_count = sum(1 for r in results if r.passed)
        total_count = len(results)
        pass_rate = (passed_count / total_count * 100) if total_count > 0 else 0

        print(f"\nOverall: {passed_count}/{total_count} tests passed ({pass_rate:.1f}%)\n")

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

    def _print_anki_summary(
        self, results: List[EvaluationResult], tool_count: int, start_time: float
    ) -> None:
        """Print summary for Anki tests."""
        total_time = (time.time() - start_time) * 1000
        passed_count = sum(1 for r in results if r.passed)
        total_count = len(results)
        pass_rate = (passed_count / total_count * 100) if total_count > 0 else 0

        print("\n" + "=" * 80)
        print("ANKI LARGE TOOLSET RESULTS SUMMARY")
        print("=" * 80)

        print(f"\n📊 Overall Statistics:")
        print(f"   Tools Available: {tool_count}")
        print(f"   Tests Passed: {passed_count}/{total_count} ({pass_rate:.1f}%)")
        print(f"   Total Time: {total_time:.1f}ms")

        if results:
            times = [r.evaluation_time_ms for r in results if r.evaluation_time_ms]
            if times:
                avg_time = sum(times) / len(times)
                print(f"   Average Test Time: {avg_time:.1f}ms")

        # Analyze by complexity
        complexity_stats = {}
        for result in results:
            for tag in result.tags:
                if tag in ["basic", "intermediate", "advanced", "expert"]:
                    if tag not in complexity_stats:
                        complexity_stats[tag] = {"passed": 0, "total": 0}
                    complexity_stats[tag]["total"] += 1
                    if result.passed:
                        complexity_stats[tag]["passed"] += 1

        if complexity_stats:
            print(f"\n📈 Performance by Complexity:")
            for complexity, stats in sorted(complexity_stats.items()):
                rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
                print(f"   {complexity.title()}: {stats['passed']}/{stats['total']} ({rate:.1f}%)")

        print(f"\n📋 Individual Results:")
        for result in results:
            status = "PASS ✓" if result.passed else "FAIL ✗"
            time_str = f"({result.evaluation_time_ms:.1f}ms)" if result.evaluation_time_ms else "(N/A)"
            complexity = next(
                (tag for tag in result.tags if tag in ["basic", "intermediate", "advanced", "expert"]),
                "other",
            )
            print(f"   {status} {result.test_id} [{complexity}] {time_str}")

            if not result.passed:
                print(f"      Expected: {[f'{tc.name}({tc.arguments})' for tc in result.expected_tool_calls]}")
                print(f"      Predicted: {[f'{tc.name}({tc.arguments})' for tc in result.predicted_tool_calls]}")
                if result.error:
                    print(f"      Error: {result.error}")

    def _print_text_summary(self, results: List) -> None:
        """Print summary for text evaluation tests."""
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

    def _print_final_summary(self, results: List[EvaluationResult], start_time: float) -> None:
        """Print final summary for all tests."""
        total_time = (time.time() - start_time) * 1000
        passed_count = sum(1 for r in results if r.passed)
        total_count = len(results)
        pass_rate = (passed_count / total_count * 100) if total_count > 0 else 0

        print("\n" + "=" * 80)
        print("FINAL RESULTS SUMMARY")
        print("=" * 80)

        print(f"\n📊 Overall Statistics:")
        print(f"   Tests Passed: {passed_count}/{total_count} ({pass_rate:.1f}%)")
        print(f"   Total Time: {total_time:.1f}ms")

        if results:
            times = [r.evaluation_time_ms for r in results if r.evaluation_time_ms]
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                print(f"   Average Test Time: {avg_time:.1f}ms")
                print(f"   Fastest Test: {min_time:.1f}ms")
                print(f"   Slowest Test: {max_time:.1f}ms")

        print(f"\n📋 Individual Results:")
        for result in results:
            status = "PASS ✓" if result.passed else "FAIL ✗"
            time_str = f"({result.evaluation_time_ms:.1f}ms)" if result.evaluation_time_ms else "(N/A)"
            print(f"   {status} {result.test_id} {time_str}")

            if not result.passed:
                print(f"      Expected: {[f'{tc.name}({tc.arguments})' for tc in result.expected_tool_calls]}")
                print(f"      Predicted: {[f'{tc.name}({tc.arguments})' for tc in result.predicted_tool_calls]}")
                if result.error:
                    print(f"      Error: {result.error}")

    def _get_custom_email_tools(self) -> List[Dict[str, Any]]:
        """Get custom email tools definition."""
        return [
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
