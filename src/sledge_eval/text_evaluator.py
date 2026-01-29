"""Text-based evaluation logic for language models."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .evaluator import (
    Evaluator,
    TextEvaluationTest,
    TextEvaluationSuite,
    TextEvaluationResult,
    EvaluationReport,
)
from .hardware_detector import HardwareDetector


class TextEvaluator(Evaluator):
    """Evaluates model performance on text-based questions."""

    def __init__(self, model_client: Any):
        """
        Initialize the text evaluator.

        Args:
            model_client: The language model client to use for evaluation
        """
        super().__init__(model_client)

    def load_text_test_suite(self, test_file: Path) -> TextEvaluationSuite:
        """
        Load a text evaluation test suite from a JSON file.

        Args:
            test_file: Path to the JSON test file

        Returns:
            TextEvaluationSuite object
        """
        with open(test_file, "r") as f:
            data = json.load(f)
        return TextEvaluationSuite(**data)

    def evaluate_text_test(self, test: TextEvaluationTest) -> TextEvaluationResult:
        """
        Evaluate a single text test case.

        Args:
            test: The text test case to evaluate

        Returns:
            TextEvaluationResult with pass/fail and details
        """
        start_time = time.time()
        
        try:
            # Get model response
            response = self._get_model_response(test.question)
            
            # Evaluate the response
            passed = self._evaluate_response(response, test.expected_answer, test.evaluation_type)
            
            evaluation_time_ms = (time.time() - start_time) * 1000
            
            return TextEvaluationResult(
                test_id=test.id,
                passed=passed,
                predicted_answer=response,
                expected_answer=test.expected_answer,
                question=test.question,
                evaluation_time_ms=evaluation_time_ms,
                test_description=test.description,
                tags=test.tags,
                evaluation_type=test.evaluation_type
            )
            
        except Exception as e:
            evaluation_time_ms = (time.time() - start_time) * 1000
            return TextEvaluationResult(
                test_id=test.id,
                passed=False,
                predicted_answer="",
                expected_answer=test.expected_answer,
                question=test.question,
                error=str(e),
                evaluation_time_ms=evaluation_time_ms,
                test_description=test.description,
                tags=test.tags,
                evaluation_type=test.evaluation_type
            )

    def _get_model_response(self, question: str) -> str:
        """
        Get response from the model for a given question.
        
        This method should be overridden by specific evaluator implementations.
        """
        raise NotImplementedError("Implement model inference logic")

    def _evaluate_response(self, predicted: str, expected: str, evaluation_type: str) -> bool:
        """
        Evaluate if the predicted response matches the expected answer.
        
        Args:
            predicted: The model's response
            expected: The expected answer
            evaluation_type: Type of evaluation (contains, exact, custom)
            
        Returns:
            True if the response passes the evaluation
        """
        predicted_clean = predicted.strip().lower()
        expected_clean = expected.strip().lower()
        
        if evaluation_type == "exact":
            return predicted_clean == expected_clean
        elif evaluation_type == "contains":
            return expected_clean in predicted_clean
        elif evaluation_type == "letter_count":
            # Special evaluation for letter counting tasks
            return self._evaluate_letter_count(predicted, expected)
        elif evaluation_type == "theory_of_mind":
            # Special evaluation for theory of mind tasks (Sally-Anne test)
            return self._evaluate_theory_of_mind(predicted, expected)
        else:
            # Default to contains
            return expected_clean in predicted_clean

    def _evaluate_letter_count(self, predicted: str, expected: str) -> bool:
        """
        Special evaluation for letter counting tasks.
        
        Looks for numbers in the response and checks if any match the expected count.
        """
        import re
        
        # Extract numbers from the predicted response
        numbers = re.findall(r'\b\d+\b', predicted)
        
        # Check if the expected number appears in the response
        expected_num = expected.strip()
        return expected_num in numbers

    def _evaluate_theory_of_mind(self, predicted: str, expected: str) -> bool:
        """
        Special evaluation for theory of mind tasks (Sally-Anne test).
        
        Evaluates whether the model correctly identifies that Sally will look in the 
        original location (basket) rather than the new location (box) where the marble 
        actually is. This tests the model's ability to understand false beliefs.
        
        Args:
            predicted: The model's response
            expected: The expected answer (should be "basket")
            
        Returns:
            True if the model correctly identifies Sally's false belief
        """
        import re
        
        predicted_lower = predicted.lower()
        expected_lower = expected.lower().strip()
        
        # Define keywords for correct and incorrect responses
        correct_indicators = [
            "basket", "original", "first", "initially", "where she left", 
            "where she put", "where she placed", "where she hid"
        ]
        
        incorrect_indicators = [
            "box", "new location", "where it is", "where the marble is",
            "where anne moved", "where anne put", "current location"
        ]
        
        # Check for explicit mention of the expected answer
        if expected_lower in predicted_lower:
            return True
            
        # Look for correct reasoning indicators
        correct_matches = sum(1 for indicator in correct_indicators if indicator in predicted_lower)
        incorrect_matches = sum(1 for indicator in incorrect_indicators if indicator in predicted_lower)
        
        # Additional reasoning checks
        reasoning_indicators = [
            "doesn't know", "doesn't see", "wasn't there", "didn't see", 
            "false belief", "her knowledge", "her perspective", "thinks it's still"
        ]
        
        has_reasoning = any(indicator in predicted_lower for indicator in reasoning_indicators)
        
        # Scoring logic:
        # - If expected answer is mentioned AND no incorrect indicators, pass
        # - If correct indicators > incorrect indicators AND has reasoning, pass
        # - If only correct indicators and no incorrect ones, pass
        
        if expected_lower in predicted_lower and incorrect_matches == 0:
            return True
        elif correct_matches > incorrect_matches and has_reasoning:
            return True
        elif correct_matches > 0 and incorrect_matches == 0:
            return True
        else:
            return False

    def evaluate_text_suite(self, test_suite: TextEvaluationSuite) -> List[TextEvaluationResult]:
        """
        Evaluate an entire text test suite.

        Args:
            test_suite: The text test suite to evaluate

        Returns:
            List of text evaluation results
        """
        results = []
        for test in test_suite.tests:
            result = self.evaluate_text_test(test)
            results.append(result)
        return results

    def create_text_evaluation_report(
        self, 
        results: List[TextEvaluationResult], 
        model_name: str,
        test_suite_name: Optional[str] = None,
        evaluation_mode: str = "text_evaluation"
    ) -> EvaluationReport:
        """
        Create an evaluation report from text evaluation results.
        
        Note: This adapts the existing EvaluationReport to work with text evaluation results.
        """
        # Convert TextEvaluationResult to EvaluationResult for compatibility
        converted_results = []
        for result in results:
            from .evaluator import EvaluationResult, ToolCall

            eval_result = EvaluationResult(
                test_id=result.test_id,
                passed=result.passed,
                predicted_tool_calls=[],  # Empty for text evaluations
                expected_tool_calls=[],   # Empty for text evaluations
                error=result.error,
                evaluation_time_ms=result.evaluation_time_ms,
                voice_command=result.question,  # Use question as voice_command
                test_description=result.test_description,
                tags=result.tags,
                predicted_answer=result.predicted_answer,
                expected_answer=result.expected_answer,
                evaluation_type=result.evaluation_type,
            )
            converted_results.append(eval_result)
        
        # Create hardware info
        hardware_detector = HardwareDetector()
        hardware_info = hardware_detector.detect()
        
        # Calculate metrics
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = total_tests - passed_tests
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        total_time = sum(r.evaluation_time_ms or 0 for r in results)
        
        # Create report
        report = EvaluationReport(
            model_name=model_name,
            evaluation_mode=evaluation_mode,
            test_suite_name=test_suite_name,
            hardware_info=hardware_info,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            pass_rate=pass_rate,
            total_evaluation_time_ms=total_time,
            test_results=converted_results
        )
        
        return report


def create_letter_counting_test_suite() -> TextEvaluationSuite:
    """
    Create a sample test suite for letter counting tasks.
    
    Returns:
        TextEvaluationSuite with letter counting tests
    """
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
            expected_answer="3",
            description="Count occurrences of letter 'e' in 'development'",
            tags=["letter_counting", "medium"],
            evaluation_type="letter_count"
        )
    ]
    
    return TextEvaluationSuite(
        name="Letter Counting Tests",
        description="Test suite for evaluating letter counting abilities",
        tests=tests
    )


def create_theory_of_mind_test_suite() -> TextEvaluationSuite:
    """
    Create a test suite for Theory of Mind tasks (Sally-Anne variations).
    
    Returns:
        TextEvaluationSuite with theory of mind tests
    """
    tests = [
        TextEvaluationTest(
            id="theory_of_mind_001",
            question="Sally hides a marble in the basket and leaves the room. Anne moves the marble to the box. Sally returns. Where will Sally look for the marble? Explain why.",
            expected_answer="basket",
            description="Classic Sally-Anne false belief test - tests understanding that Sally doesn't know the marble was moved",
            tags=["theory_of_mind", "false_belief", "perspective_taking", "advanced"],
            evaluation_type="theory_of_mind"
        ),
        TextEvaluationTest(
            id="theory_of_mind_002",
            question="John puts his chocolate in the cupboard and goes to school. His mother moves the chocolate to the fridge. When John comes home, where will he first look for his chocolate?",
            expected_answer="cupboard",
            description="Variation of false belief test with different objects and actors",
            tags=["theory_of_mind", "false_belief", "perspective_taking", "advanced"],
            evaluation_type="theory_of_mind"
        ),
        TextEvaluationTest(
            id="theory_of_mind_003",
            question="Maria places her book on the desk and leaves for lunch. While she's away, Tom moves the book to the shelf. When Maria returns, where will she think her book is?",
            expected_answer="desk",
            description="Another false belief variation testing understanding of another person's outdated knowledge",
            tags=["theory_of_mind", "false_belief", "perspective_taking", "advanced"],
            evaluation_type="theory_of_mind"
        )
    ]
    
    return TextEvaluationSuite(
        name="Theory of Mind Tests",
        description="Test suite for evaluating theory of mind and false belief understanding (Sally-Anne variations)",
        tests=tests
    )


def create_comprehensive_text_test_suite() -> TextEvaluationSuite:
    """
    Create a comprehensive test suite combining multiple text evaluation types.
    
    Returns:
        TextEvaluationSuite with letter counting and theory of mind tests
    """
    # Get tests from specialized suites
    letter_tests = create_letter_counting_test_suite().tests
    theory_tests = create_theory_of_mind_test_suite().tests
    
    # Combine all tests
    all_tests = letter_tests + theory_tests
    
    return TextEvaluationSuite(
        name="Comprehensive Text Evaluation Suite",
        description="Combined test suite evaluating letter counting abilities and theory of mind understanding",
        tests=all_tests
    )