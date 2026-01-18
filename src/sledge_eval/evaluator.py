"""Core evaluation logic for voice command interpretation."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from .hardware_detector import HardwareInfo, HardwareDetector


class ToolCall(BaseModel):
    """Represents a tool call with function name and arguments."""

    name: str = Field(..., description="The name of the tool/function to call")
    arguments: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments to pass to the tool"
    )


class VoiceCommandTest(BaseModel):
    """Represents a single voice command test case."""

    id: str = Field(..., description="Unique identifier for the test case")
    voice_command: str = Field(..., description="The voice command text")
    expected_tool_calls: List[ToolCall] = Field(
        ..., description="Expected tool calls that should be generated"
    )
    description: Optional[str] = Field(
        None, description="Optional description of what this test validates"
    )
    tags: List[str] = Field(
        default_factory=list, description="Tags for categorizing tests"
    )


class TextEvaluationTest(BaseModel):
    """Represents a single text evaluation test case."""

    id: str = Field(..., description="Unique identifier for the test case")
    question: str = Field(..., description="The question to ask the model")
    expected_answer: str = Field(..., description="Expected text answer")
    description: Optional[str] = Field(
        None, description="Optional description of what this test validates"
    )
    tags: List[str] = Field(
        default_factory=list, description="Tags for categorizing tests"
    )
    evaluation_type: str = Field(
        default="contains", description="Type of evaluation: contains, exact, custom"
    )


class TestSuite(BaseModel):
    """A collection of voice command tests."""

    name: str = Field(..., description="Name of the test suite")
    description: Optional[str] = Field(None, description="Test suite description")
    tests: List[VoiceCommandTest] = Field(
        default_factory=list, description="List of test cases"
    )


class TextEvaluationSuite(BaseModel):
    """A collection of text evaluation tests."""

    name: str = Field(..., description="Name of the test suite")
    description: Optional[str] = Field(None, description="Test suite description")
    tests: List[TextEvaluationTest] = Field(
        default_factory=list, description="List of test cases"
    )


class EvaluationResult(BaseModel):
    """Result of evaluating a single test case."""

    test_id: str
    passed: bool
    predicted_tool_calls: List[ToolCall]
    expected_tool_calls: List[ToolCall]
    error: Optional[str] = None
    evaluation_time_ms: Optional[float] = Field(
        None, description="Time taken for evaluation in milliseconds"
    )
    voice_command: Optional[str] = None
    test_description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class TextEvaluationResult(BaseModel):
    """Result of evaluating a single text evaluation test case."""

    test_id: str
    passed: bool
    predicted_answer: str
    expected_answer: str
    question: str
    error: Optional[str] = None
    evaluation_time_ms: Optional[float] = Field(
        None, description="Time taken for evaluation in milliseconds"
    )
    test_description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    evaluation_type: str = Field(default="contains")


class EvaluationReport(BaseModel):
    """Comprehensive report of an evaluation session."""

    # Session metadata
    timestamp: datetime = Field(default_factory=datetime.now)
    model_name: str
    server_url: Optional[str] = None
    evaluation_mode: str  # single, suite, custom, all
    test_suite_name: Optional[str] = None
    
    # Hardware information
    hardware_info: Optional[HardwareInfo] = None
    
    # Results summary
    total_tests: int
    passed_tests: int
    failed_tests: int
    pass_rate: float = Field(description="Percentage of tests passed (0-100)")
    total_evaluation_time_ms: float
    
    # Detailed results
    test_results: List[EvaluationResult]
    
    # Error tracking
    errors: List[str] = Field(default_factory=list)
    
    # Tags and categories
    tags_tested: List[str] = Field(default_factory=list)
    tag_performance: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    @property
    def display_name(self) -> str:
        """Get a clean display name for the model, extracting basename for local paths."""
        name = self.model_name.strip('"').strip("'")
        if name.endswith('.gguf') and '/' in name:
            return Path(name).stem
        return name

    def add_result(self, result: EvaluationResult):
        """Add a test result to the report."""
        self.test_results.append(result)
        self.total_tests += 1
        
        if result.passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
            
        if result.error:
            self.errors.append(f"Test {result.test_id}: {result.error}")
            
        # Track tags
        for tag in result.tags:
            if tag not in self.tags_tested:
                self.tags_tested.append(tag)
            
            if tag not in self.tag_performance:
                self.tag_performance[tag] = {"passed": 0, "failed": 0, "total": 0}
            
            self.tag_performance[tag]["total"] += 1
            if result.passed:
                self.tag_performance[tag]["passed"] += 1
            else:
                self.tag_performance[tag]["failed"] += 1
        
        # Update pass rate
        self.pass_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
    
    def save_to_file(self, base_path: Path):
        """Save report to JSON file in the specified directory structure."""
        # Create directory structure: /reports/model_name/
        # For local file paths, extract just the filename
        model_name = self.model_name.strip('"').strip("'")
        if model_name.endswith('.gguf') and '/' in model_name:
            model_name = Path(model_name).stem  # Get filename without extension

        # Clean model name for safe filesystem use
        clean_model_name = (model_name
                          .replace("/", "_")
                          .replace(":", "_")
                          .replace('"', "")
                          .replace("'", "")
                          .replace(" ", "_"))
        model_dir = base_path / "reports" / clean_model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filenames with timestamp
        timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
        json_filename = f"report_{timestamp_str}.json"
        md_filename = f"report_{timestamp_str}.md"
        json_filepath = model_dir / json_filename
        md_filepath = model_dir / md_filename
        
        # Save JSON report
        with open(json_filepath, 'w') as f:
            json.dump(self.model_dump(), f, indent=2, default=str)
        
        # Generate and save markdown report
        markdown_content = self.generate_markdown()
        with open(md_filepath, 'w') as f:
            f.write(markdown_content)
        
        return {"json": json_filepath, "markdown": md_filepath}
    
    def generate_markdown(self) -> str:
        """Generate a comprehensive markdown report.

        Delegates to MarkdownRenderer for actual rendering.
        """
        from .reporting.markdown_renderer import MarkdownRenderer

        renderer = MarkdownRenderer()
        return renderer.render(self)


class Evaluator:
    """Evaluates model performance on voice command to tool call tasks."""

    def __init__(self, model_client: Any):
        """
        Initialize the evaluator.

        Args:
            model_client: The language model client to use for evaluation
        """
        self.model_client = model_client

    def _compare_tool_calls(
        self, predicted: List[ToolCall], expected: List[ToolCall]
    ) -> bool:
        """
        Compare predicted and expected tool calls.

        Args:
            predicted: List of predicted tool calls
            expected: List of expected tool calls

        Returns:
            True if they match, False otherwise
        """
        from .utils.comparison import compare_tool_calls
        return compare_tool_calls(predicted, expected, strict=False)

    def load_test_suite(self, test_file: Path) -> TestSuite:
        """
        Load a test suite from a JSON file.

        Args:
            test_file: Path to the JSON test file

        Returns:
            TestSuite object
        """
        with open(test_file, "r") as f:
            data = json.load(f)
        return TestSuite(**data)

    def evaluate_test(self, test: VoiceCommandTest) -> EvaluationResult:
        """
        Evaluate a single test case.

        Args:
            test: The test case to evaluate

        Returns:
            EvaluationResult with pass/fail and details
        """
        # This is a placeholder - implement actual model inference here
        raise NotImplementedError("Implement model inference logic")

    def evaluate_suite(self, test_suite: TestSuite) -> List[EvaluationResult]:
        """
        Evaluate an entire test suite.

        Args:
            test_suite: The test suite to evaluate

        Returns:
            List of evaluation results
        """
        results = []
        for test in test_suite.tests:
            result = self.evaluate_test(test)
            results.append(result)
        return results
