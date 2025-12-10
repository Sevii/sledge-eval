"""Core evaluation logic for voice command interpretation."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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


class TestSuite(BaseModel):
    """A collection of voice command tests."""

    name: str = Field(..., description="Name of the test suite")
    description: Optional[str] = Field(None, description="Test suite description")
    tests: List[VoiceCommandTest] = Field(
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


class EvaluationReport(BaseModel):
    """Comprehensive report of an evaluation session."""

    # Session metadata
    timestamp: datetime = Field(default_factory=datetime.now)
    model_name: str
    server_url: Optional[str] = None
    evaluation_mode: str  # single, suite, custom, all
    test_suite_name: Optional[str] = None
    
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
        # Clean model name for safe filesystem use
        clean_model_name = (self.model_name
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
        """Generate a comprehensive markdown report."""
        lines = []
        
        # Header
        lines.append(f"# Evaluation Report: {self.model_name}")
        lines.append("")
        lines.append(f"**Generated:** {self.timestamp.strftime('%B %d, %Y at %I:%M:%S %p')}")
        lines.append(f"**Model:** `{self.model_name}`")
        lines.append(f"**Server URL:** {self.server_url or 'N/A'}")
        lines.append(f"**Evaluation Mode:** {self.evaluation_mode}")
        if self.test_suite_name:
            lines.append(f"**Test Suite:** {self.test_suite_name}")
        lines.append("")
        
        # Executive Summary
        lines.append("## 📊 Executive Summary")
        lines.append("")
        
        # Pass rate with emoji
        if self.pass_rate >= 90:
            status_emoji = "🟢"
        elif self.pass_rate >= 70:
            status_emoji = "🟡"
        else:
            status_emoji = "🔴"
        
        lines.append(f"- **Overall Score:** {status_emoji} {self.pass_rate:.1f}% ({self.passed_tests}/{self.total_tests} tests passed)")
        lines.append(f"- **Total Evaluation Time:** {self.total_evaluation_time_ms:.1f}ms")
        
        if self.test_results:
            avg_time = sum(r.evaluation_time_ms or 0 for r in self.test_results) / len(self.test_results)
            lines.append(f"- **Average Test Time:** {avg_time:.1f}ms")
        
        lines.append("")
        
        # Performance by Tag
        if self.tag_performance:
            lines.append("## 🏷️ Performance by Category")
            lines.append("")
            lines.append("| Category | Passed | Failed | Total | Pass Rate |")
            lines.append("|----------|--------|--------|-------|-----------|")
            
            for tag, perf in sorted(self.tag_performance.items()):
                pass_rate = (perf["passed"] / perf["total"] * 100) if perf["total"] > 0 else 0
                if pass_rate >= 90:
                    emoji = "🟢"
                elif pass_rate >= 70:
                    emoji = "🟡"
                else:
                    emoji = "🔴"
                lines.append(f"| {emoji} {tag} | {perf['passed']} | {perf['failed']} | {perf['total']} | {pass_rate:.1f}% |")
            lines.append("")
        
        # Detailed Test Results
        lines.append("## 📋 Detailed Test Results")
        lines.append("")
        
        # Group by pass/fail
        passed_tests = [r for r in self.test_results if r.passed]
        failed_tests = [r for r in self.test_results if not r.passed]
        
        if passed_tests:
            lines.append("### ✅ Passed Tests")
            lines.append("")
            for result in passed_tests:
                lines.append(f"#### {result.test_id}")
                lines.append("")
                if result.voice_command:
                    lines.append(f"**Command:** `{result.voice_command}`")
                if result.test_description:
                    lines.append(f"**Description:** {result.test_description}")
                if result.tags:
                    lines.append(f"**Tags:** {', '.join(result.tags)}")
                if result.evaluation_time_ms:
                    lines.append(f"**Execution Time:** {result.evaluation_time_ms:.1f}ms")
                
                lines.append("")
                lines.append("**Expected Tool Calls:**")
                for tc in result.expected_tool_calls:
                    lines.append(f"- `{tc.name}({tc.arguments})`")
                
                lines.append("")
                lines.append("**Predicted Tool Calls:**")
                for tc in result.predicted_tool_calls:
                    lines.append(f"- `{tc.name}({tc.arguments})`")
                
                lines.append("")
        
        if failed_tests:
            lines.append("### ❌ Failed Tests")
            lines.append("")
            for result in failed_tests:
                lines.append(f"#### {result.test_id}")
                lines.append("")
                if result.voice_command:
                    lines.append(f"**Command:** `{result.voice_command}`")
                if result.test_description:
                    lines.append(f"**Description:** {result.test_description}")
                if result.tags:
                    lines.append(f"**Tags:** {', '.join(result.tags)}")
                if result.evaluation_time_ms:
                    lines.append(f"**Execution Time:** {result.evaluation_time_ms:.1f}ms")
                
                if result.error:
                    lines.append("")
                    lines.append(f"**Error:** `{result.error}`")
                
                lines.append("")
                lines.append("**Expected Tool Calls:**")
                for tc in result.expected_tool_calls:
                    lines.append(f"- `{tc.name}({tc.arguments})`")
                
                lines.append("")
                lines.append("**Predicted Tool Calls:**")
                if result.predicted_tool_calls:
                    for tc in result.predicted_tool_calls:
                        lines.append(f"- `{tc.name}({tc.arguments})`")
                else:
                    lines.append("- *No tool calls predicted*")
                
                lines.append("")
        
        # Error Summary
        if self.errors:
            lines.append("## ⚠️ Errors Encountered")
            lines.append("")
            for i, error in enumerate(self.errors, 1):
                lines.append(f"{i}. `{error}`")
            lines.append("")
        
        # Technical Details
        lines.append("## 🔧 Technical Details")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Model Name | `{self.model_name}` |")
        lines.append(f"| Server URL | {self.server_url or 'N/A'} |")
        lines.append(f"| Evaluation Mode | {self.evaluation_mode} |")
        lines.append(f"| Timestamp | {self.timestamp.isoformat()} |")
        lines.append(f"| Total Tests | {self.total_tests} |")
        lines.append(f"| Passed Tests | {self.passed_tests} |")
        lines.append(f"| Failed Tests | {self.failed_tests} |")
        lines.append(f"| Pass Rate | {self.pass_rate:.2f}% |")
        lines.append(f"| Total Time | {self.total_evaluation_time_ms:.1f}ms |")
        lines.append("")
        
        # Footer
        lines.append("---")
        lines.append("*This report was automatically generated by Sledge Eval*")
        
        return "\n".join(lines)


class Evaluator:
    """Evaluates model performance on voice command to tool call tasks."""

    def __init__(self, model_client: Any):
        """
        Initialize the evaluator.

        Args:
            model_client: The language model client to use for evaluation
        """
        self.model_client = model_client

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
