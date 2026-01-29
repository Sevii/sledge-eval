"""Latency-focused evaluator for benchmarking inference speed optimizations."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from pydantic import BaseModel

from .evaluator import Evaluator, ToolCall, VoiceCommandTest, EvaluationResult
from .tools.defaults import get_default_tools


class LatencyMetrics(BaseModel):
    """Detailed latency metrics for a single evaluation."""

    # Timing breakdown (all in milliseconds)
    total_latency_ms: float
    ttft_ms: Optional[float] = None  # Time to first token
    generation_time_ms: Optional[float] = None  # Time after first token

    # Token metrics
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    expected_output_tokens: Optional[int] = None

    # Throughput
    tokens_per_second: Optional[float] = None
    ms_per_token: Optional[float] = None

    # Target comparison
    target_latency_ms: float = 100.0
    latency_gap_ms: Optional[float] = None  # How far from target
    meets_target: bool = False

    # Optimization metadata
    optimization_category: Optional[str] = None
    prefix_used: Optional[str] = None

    def calculate_derived_metrics(self):
        """Calculate derived metrics from raw measurements."""
        if self.total_latency_ms is not None:
            self.latency_gap_ms = self.total_latency_ms - self.target_latency_ms
            self.meets_target = self.total_latency_ms <= self.target_latency_ms

        if self.output_tokens and self.generation_time_ms and self.generation_time_ms > 0:
            self.tokens_per_second = (self.output_tokens / self.generation_time_ms) * 1000
            self.ms_per_token = self.generation_time_ms / self.output_tokens


class LatencyTestResult(BaseModel):
    """Extended result for latency benchmarking."""

    test_id: str
    passed: bool
    voice_command: str
    category: Optional[str] = None

    # Tool call results
    predicted_tool_calls: List[ToolCall] = []
    expected_tool_calls: List[ToolCall] = []
    tool_call_correct: bool = False

    # Latency metrics
    latency: LatencyMetrics

    # Error handling
    error: Optional[str] = None

    # Test metadata
    description: Optional[str] = None
    tags: List[str] = []


class LatencyBenchmarkReport(BaseModel):
    """Aggregated report for latency benchmarks."""

    model_name: str
    server_url: Optional[str] = None
    timestamp: str

    # Overall metrics
    total_tests: int = 0
    tests_meeting_target: int = 0
    target_hit_rate: float = 0.0

    # Latency statistics
    target_latency_ms: float = 100.0
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Throughput statistics
    avg_tokens_per_second: float = 0.0
    max_tokens_per_second: float = 0.0

    # TTFT statistics
    avg_ttft_ms: Optional[float] = None
    min_ttft_ms: Optional[float] = None
    max_ttft_ms: Optional[float] = None

    # Results by category
    results_by_category: Dict[str, Dict[str, Any]] = {}

    # Individual results
    test_results: List[LatencyTestResult] = []

    # Tool call accuracy
    tool_call_accuracy: float = 0.0

    def calculate_statistics(self):
        """Calculate aggregate statistics from individual results."""
        if not self.test_results:
            return

        self.total_tests = len(self.test_results)

        # Collect latencies
        latencies = [r.latency.total_latency_ms for r in self.test_results]
        latencies.sort()

        # Basic stats
        self.avg_latency_ms = sum(latencies) / len(latencies)
        self.min_latency_ms = min(latencies)
        self.max_latency_ms = max(latencies)

        # Percentiles
        self.p50_latency_ms = self._percentile(latencies, 50)
        self.p95_latency_ms = self._percentile(latencies, 95)
        self.p99_latency_ms = self._percentile(latencies, 99)

        # Target metrics
        self.tests_meeting_target = sum(1 for r in self.test_results if r.latency.meets_target)
        self.target_hit_rate = (self.tests_meeting_target / self.total_tests) * 100

        # Throughput
        throughputs = [r.latency.tokens_per_second for r in self.test_results
                      if r.latency.tokens_per_second is not None]
        if throughputs:
            self.avg_tokens_per_second = sum(throughputs) / len(throughputs)
            self.max_tokens_per_second = max(throughputs)

        # TTFT stats
        ttfts = [r.latency.ttft_ms for r in self.test_results
                if r.latency.ttft_ms is not None]
        if ttfts:
            self.avg_ttft_ms = sum(ttfts) / len(ttfts)
            self.min_ttft_ms = min(ttfts)
            self.max_ttft_ms = max(ttfts)

        # Tool call accuracy
        correct = sum(1 for r in self.test_results if r.tool_call_correct)
        self.tool_call_accuracy = (correct / self.total_tests) * 100

        # Results by category
        categories: Dict[str, List[LatencyTestResult]] = {}
        for result in self.test_results:
            cat = result.category or "uncategorized"
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result)

        for cat, results in categories.items():
            cat_latencies = [r.latency.total_latency_ms for r in results]
            cat_meeting_target = sum(1 for r in results if r.latency.meets_target)
            self.results_by_category[cat] = {
                "count": len(results),
                "avg_latency_ms": sum(cat_latencies) / len(cat_latencies),
                "min_latency_ms": min(cat_latencies),
                "max_latency_ms": max(cat_latencies),
                "meeting_target": cat_meeting_target,
                "target_rate": (cat_meeting_target / len(results)) * 100,
            }

    def _percentile(self, sorted_data: List[float], percentile: int) -> float:
        """Calculate percentile from sorted data."""
        if not sorted_data:
            return 0.0
        k = (len(sorted_data) - 1) * percentile / 100
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_data) else f
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])

    def generate_markdown(self) -> str:
        """Generate markdown report."""
        lines = []

        lines.append(f"# Latency Benchmark Report: {self.model_name}")
        lines.append("")
        lines.append(f"**Target Latency:** {self.target_latency_ms}ms")
        lines.append(f"**Timestamp:** {self.timestamp}")
        if self.server_url:
            lines.append(f"**Server:** {self.server_url}")
        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")

        target_emoji = "🎯" if self.target_hit_rate >= 50 else "⚠️"
        lines.append(f"- **Target Hit Rate:** {target_emoji} {self.target_hit_rate:.1f}% ({self.tests_meeting_target}/{self.total_tests} tests)")
        lines.append(f"- **Average Latency:** {self.avg_latency_ms:.1f}ms (target: {self.target_latency_ms}ms)")
        lines.append(f"- **Latency Range:** {self.min_latency_ms:.1f}ms - {self.max_latency_ms:.1f}ms")
        lines.append(f"- **Tool Call Accuracy:** {self.tool_call_accuracy:.1f}%")
        lines.append("")

        # Latency Distribution
        lines.append("## Latency Distribution")
        lines.append("")
        lines.append("| Percentile | Latency (ms) |")
        lines.append("|------------|--------------|")
        lines.append(f"| P50 | {self.p50_latency_ms:.1f} |")
        lines.append(f"| P95 | {self.p95_latency_ms:.1f} |")
        lines.append(f"| P99 | {self.p99_latency_ms:.1f} |")
        lines.append("")

        # Throughput
        if self.avg_tokens_per_second > 0:
            lines.append("## Throughput")
            lines.append("")
            lines.append(f"- **Average:** {self.avg_tokens_per_second:.1f} tokens/sec")
            lines.append(f"- **Peak:** {self.max_tokens_per_second:.1f} tokens/sec")

            # Calculate if we're hitting the 375 t/s target
            target_tps = 375
            if self.avg_tokens_per_second >= target_tps:
                lines.append(f"- **Status:** ✅ Meeting target ({target_tps} t/s)")
            else:
                gap = target_tps - self.avg_tokens_per_second
                lines.append(f"- **Status:** ❌ Below target by {gap:.1f} t/s")
            lines.append("")

        # TTFT
        if self.avg_ttft_ms is not None:
            lines.append("## Time to First Token (TTFT)")
            lines.append("")
            lines.append(f"- **Average:** {self.avg_ttft_ms:.1f}ms")
            lines.append(f"- **Range:** {self.min_ttft_ms:.1f}ms - {self.max_ttft_ms:.1f}ms")
            lines.append("")

        # Results by Category
        if self.results_by_category:
            lines.append("## Results by Optimization Category")
            lines.append("")
            lines.append("| Category | Tests | Avg Latency | Target Rate | Improvement |")
            lines.append("|----------|-------|-------------|-------------|-------------|")

            baseline_latency = self.results_by_category.get("baseline", {}).get("avg_latency_ms", self.avg_latency_ms)

            for cat, stats in sorted(self.results_by_category.items()):
                improvement = ((baseline_latency - stats["avg_latency_ms"]) / baseline_latency) * 100 if baseline_latency > 0 else 0
                imp_str = f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"
                if cat == "baseline":
                    imp_str = "-"

                rate_emoji = "✅" if stats["target_rate"] >= 50 else "❌"
                lines.append(f"| {cat} | {stats['count']} | {stats['avg_latency_ms']:.1f}ms | {rate_emoji} {stats['target_rate']:.1f}% | {imp_str} |")
            lines.append("")

        # Individual Results
        lines.append("## Individual Test Results")
        lines.append("")
        lines.append("| Test ID | Category | Latency | Target | Tool Correct |")
        lines.append("|---------|----------|---------|--------|--------------|")

        for result in self.test_results:
            target_status = "✅" if result.latency.meets_target else "❌"
            tool_status = "✅" if result.tool_call_correct else "❌"
            lines.append(f"| {result.test_id} | {result.category or '-'} | {result.latency.total_latency_ms:.1f}ms | {target_status} | {tool_status} |")
        lines.append("")

        # Recommendations
        lines.append("## Optimization Recommendations")
        lines.append("")

        if self.avg_latency_ms > self.target_latency_ms:
            gap = self.avg_latency_ms - self.target_latency_ms
            lines.append(f"Current average latency is **{gap:.1f}ms above target**. Consider:")
            lines.append("")

            if self.avg_tokens_per_second < 375:
                lines.append("1. **Increase Throughput:** Switch to TensorRT-LLM or SGLang")
                lines.append("2. **Enable Speculative Decoding:** Use a draft model for predictable tool calls")

            if "prefix_injection" in self.results_by_category:
                prefix_stats = self.results_by_category["prefix_injection"]
                if prefix_stats["avg_latency_ms"] < self.avg_latency_ms:
                    savings = self.avg_latency_ms - prefix_stats["avg_latency_ms"]
                    lines.append(f"3. **Use Prefix Injection:** Saves ~{savings:.1f}ms per call")

            if "short_fields" in self.results_by_category:
                short_stats = self.results_by_category["short_fields"]
                if short_stats["avg_latency_ms"] < self.avg_latency_ms:
                    savings = self.avg_latency_ms - short_stats["avg_latency_ms"]
                    lines.append(f"4. **Use Short Field Names:** Saves ~{savings:.1f}ms per call")

            lines.append("5. **Quantization:** Try AWQ/GPTQ Int4 to increase memory bandwidth")
        else:
            lines.append("✅ **Target achieved!** Current optimizations are working well.")

        lines.append("")
        lines.append("---")
        lines.append("*Generated by Sledge Eval Latency Benchmark*")

        return "\n".join(lines)


class LatencyBenchmarkTest(BaseModel):
    """Test case with latency-specific metadata."""

    id: str
    category: str = "baseline"
    voice_command: str
    expected_tool_calls: List[ToolCall]
    description: Optional[str] = None
    tags: List[str] = []

    # Latency-specific fields
    system_prefix: Optional[str] = None  # For prefix injection
    expected_token_count: Optional[int] = None
    optimization_savings_tokens: Optional[int] = None
    field_mapping: Optional[Dict[str, str]] = None  # For short field names
    speculative_notes: Optional[str] = None


class LatencyBenchmarkSuite(BaseModel):
    """Suite of latency benchmark tests."""

    name: str
    description: Optional[str] = None
    version: str = "1.0.0"
    target_latency_ms: float = 100.0
    metadata: Dict[str, Any] = {}
    tests: List[LatencyBenchmarkTest]


class LatencyEvaluator(Evaluator):
    """Evaluator focused on latency benchmarking with detailed timing metrics."""

    def __init__(
        self,
        server_url: str = "http://localhost:8080",
        available_tools: Optional[List[Dict[str, Any]]] = None,
        timeout: int = 120,
        debug: bool = False,
        target_latency_ms: float = 100.0,
        measure_ttft: bool = True,
    ):
        """
        Initialize the latency evaluator.

        Args:
            server_url: URL of the llama-server instance
            available_tools: List of tool definitions
            timeout: Request timeout in seconds
            debug: Enable debug logging
            target_latency_ms: Target latency to measure against
            measure_ttft: Whether to measure time to first token (requires streaming)
        """
        self.server_url = server_url.rstrip('/')
        self.available_tools = available_tools or get_default_tools()
        self.timeout = timeout
        self.debug = debug
        self.target_latency_ms = target_latency_ms
        self.measure_ttft = measure_ttft

        super().__init__(model_client=None)

    def load_latency_suite(self, test_file: Path) -> LatencyBenchmarkSuite:
        """Load a latency benchmark suite from JSON file."""
        with open(test_file, "r") as f:
            data = json.load(f)

        # Convert tool calls
        for test in data.get("tests", []):
            test["expected_tool_calls"] = [
                ToolCall(**tc) for tc in test.get("expected_tool_calls", [])
            ]

        return LatencyBenchmarkSuite(**data)

    def evaluate_latency_test(
        self,
        test: LatencyBenchmarkTest,
        warmup: bool = False,
    ) -> LatencyTestResult:
        """
        Evaluate a single latency benchmark test.

        Args:
            test: The test case to evaluate
            warmup: If True, this is a warmup run (not counted in results)

        Returns:
            LatencyTestResult with detailed timing metrics
        """
        start_time = time.perf_counter()
        ttft = None
        generation_start = None

        try:
            # Build request payload
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that interprets voice commands and calls appropriate functions. Respond only with the tool call JSON.",
                },
                {"role": "user", "content": test.voice_command},
            ]

            # Add prefix injection if specified
            if test.system_prefix:
                messages.append({
                    "role": "assistant",
                    "content": test.system_prefix,
                })

            payload = {
                "messages": messages,
                "tools": self.available_tools,
                "tool_choice": "auto",
                "temperature": 0.1,
                "max_tokens": 512,
            }

            # Make request with timing
            request_start = time.perf_counter()

            if self.measure_ttft:
                # Use streaming to measure TTFT
                payload["stream"] = True
                response = requests.post(
                    f"{self.server_url}/v1/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout,
                    stream=True,
                )

                first_chunk = True
                full_response = ""

                for line in response.iter_lines():
                    if first_chunk and line:
                        ttft = (time.perf_counter() - request_start) * 1000
                        generation_start = time.perf_counter()
                        first_chunk = False

                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]
                            if data_str.strip() == '[DONE]':
                                break
                            try:
                                chunk_data = json.loads(data_str)
                                if 'choices' in chunk_data:
                                    delta = chunk_data['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        full_response += delta['content']
                            except json.JSONDecodeError:
                                pass

                generation_time = (time.perf_counter() - generation_start) * 1000 if generation_start else None
            else:
                # Non-streaming request
                response = requests.post(
                    f"{self.server_url}/v1/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout,
                )
                generation_time = None

            total_latency = (time.perf_counter() - start_time) * 1000

            # Parse response
            predicted_tool_calls = []
            output_tokens = None

            if not self.measure_ttft:
                if response.status_code != 200:
                    raise Exception(f"Server returned {response.status_code}: {response.text}")

                response_data = response.json()

                # Get token counts if available
                if "usage" in response_data:
                    output_tokens = response_data["usage"].get("completion_tokens")

                # Parse tool calls
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    choice = response_data["choices"][0]
                    message = choice.get("message", {})

                    if "tool_calls" in message:
                        for tool_call in message["tool_calls"]:
                            if tool_call.get("type") == "function":
                                function = tool_call.get("function", {})
                                function_name = function.get("name", "")

                                arguments_str = function.get("arguments", "{}")
                                try:
                                    arguments = json.loads(arguments_str)
                                except json.JSONDecodeError:
                                    arguments = {}

                                predicted_tool_calls.append(
                                    ToolCall(name=function_name, arguments=arguments)
                                )

            # Compare tool calls
            tool_call_correct = self._compare_tool_calls(
                predicted_tool_calls, test.expected_tool_calls
            )

            # Create latency metrics
            metrics = LatencyMetrics(
                total_latency_ms=total_latency,
                ttft_ms=ttft,
                generation_time_ms=generation_time,
                output_tokens=output_tokens,
                expected_output_tokens=test.expected_token_count,
                target_latency_ms=self.target_latency_ms,
                optimization_category=test.category,
                prefix_used=test.system_prefix,
            )
            metrics.calculate_derived_metrics()

            return LatencyTestResult(
                test_id=test.id,
                passed=tool_call_correct and metrics.meets_target,
                voice_command=test.voice_command,
                category=test.category,
                predicted_tool_calls=predicted_tool_calls,
                expected_tool_calls=test.expected_tool_calls,
                tool_call_correct=tool_call_correct,
                latency=metrics,
                description=test.description,
                tags=test.tags,
            )

        except Exception as e:
            total_latency = (time.perf_counter() - start_time) * 1000

            metrics = LatencyMetrics(
                total_latency_ms=total_latency,
                target_latency_ms=self.target_latency_ms,
                optimization_category=test.category,
            )

            return LatencyTestResult(
                test_id=test.id,
                passed=False,
                voice_command=test.voice_command,
                category=test.category,
                expected_tool_calls=test.expected_tool_calls,
                latency=metrics,
                error=str(e),
                description=test.description,
                tags=test.tags,
            )

    def run_benchmark(
        self,
        suite: LatencyBenchmarkSuite,
        model_name: str,
        warmup_runs: int = 2,
        iterations: int = 1,
    ) -> LatencyBenchmarkReport:
        """
        Run the full latency benchmark suite.

        Args:
            suite: The benchmark suite to run
            model_name: Name of the model being benchmarked
            warmup_runs: Number of warmup iterations before real measurements
            iterations: Number of times to run each test (for averaging)

        Returns:
            LatencyBenchmarkReport with aggregate statistics
        """
        from datetime import datetime

        print(f"\n{'='*60}")
        print(f"LATENCY BENCHMARK: {suite.name}")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Target Latency: {suite.target_latency_ms}ms")
        print(f"Tests: {len(suite.tests)}")
        print(f"Warmup Runs: {warmup_runs}")
        print(f"Iterations: {iterations}")
        print()

        # Warmup
        if warmup_runs > 0 and len(suite.tests) > 0:
            print("Running warmup...")
            warmup_test = suite.tests[0]
            for i in range(warmup_runs):
                self.evaluate_latency_test(warmup_test, warmup=True)
            print(f"Warmup complete ({warmup_runs} runs)")
            print()

        # Run tests
        all_results: List[LatencyTestResult] = []

        for test in suite.tests:
            print(f"🧪 {test.id} [{test.category}]")
            print(f"   Command: '{test.voice_command}'")

            # Run multiple iterations and take the best result
            iteration_results = []
            for _ in range(iterations):
                result = self.evaluate_latency_test(test)
                iteration_results.append(result)

            # Use best latency result
            best_result = min(iteration_results, key=lambda r: r.latency.total_latency_ms)
            all_results.append(best_result)

            # Print result
            status = "✅" if best_result.latency.meets_target else "❌"
            tool_status = "✓" if best_result.tool_call_correct else "✗"
            print(f"   Latency: {best_result.latency.total_latency_ms:.1f}ms {status}")
            print(f"   Tool Call: {tool_status}")

            if best_result.latency.ttft_ms:
                print(f"   TTFT: {best_result.latency.ttft_ms:.1f}ms")

            if best_result.error:
                print(f"   Error: {best_result.error}")
            print()

        # Create report
        report = LatencyBenchmarkReport(
            model_name=model_name,
            server_url=self.server_url,
            timestamp=datetime.now().isoformat(),
            target_latency_ms=suite.target_latency_ms,
            test_results=all_results,
        )
        report.calculate_statistics()

        # Print summary
        print(f"{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print(f"Target Hit Rate: {report.target_hit_rate:.1f}%")
        print(f"Average Latency: {report.avg_latency_ms:.1f}ms")
        print(f"P50 Latency: {report.p50_latency_ms:.1f}ms")
        print(f"P95 Latency: {report.p95_latency_ms:.1f}ms")
        print(f"Tool Call Accuracy: {report.tool_call_accuracy:.1f}%")

        if report.avg_tokens_per_second > 0:
            print(f"Avg Throughput: {report.avg_tokens_per_second:.1f} t/s")

        print()

        return report

    def get_short_field_tools(self) -> List[Dict[str, Any]]:
        """Get tool definitions with short field names for latency optimization."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "lt",
                    "description": "Control lights",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "r": {"type": "string", "description": "Room"},
                            "a": {"type": "string", "enum": ["on", "off", "dim"]},
                        },
                        "required": ["a"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "tmp",
                    "description": "Set temperature",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "t": {"type": "number"},
                        },
                        "required": ["t"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "wx",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "l": {"type": "string"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "mus",
                    "description": "Play music",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "p": {"type": "string"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "vol",
                    "description": "Adjust volume",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "string", "enum": ["up", "down", "mute"]},
                        },
                        "required": ["a"],
                    },
                },
            },
        ]
