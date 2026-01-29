"""Latency benchmark runner for CLI."""

from pathlib import Path
from typing import Optional

from ..latency_evaluator import LatencyEvaluator, LatencyBenchmarkSuite


# Default latency test suite path
DEFAULT_LATENCY_SUITE = Path("tests/test_data/latency_benchmark_suite.json")


class LatencyBenchmarkRunner:
    """Runner for latency benchmarks."""

    def __init__(
        self,
        server_url: str,
        model_name: Optional[str] = None,
        timeout: int = 120,
        debug: bool = False,
        target_latency_ms: float = 100.0,
        measure_ttft: bool = False,  # Disabled by default as it requires streaming support
    ):
        """
        Initialize the latency benchmark runner.

        Args:
            server_url: URL of the llama-server instance
            model_name: Name of the model being benchmarked
            timeout: Request timeout in seconds
            debug: Enable debug logging
            target_latency_ms: Target latency in milliseconds
            measure_ttft: Whether to measure time to first token (requires streaming)
        """
        self.server_url = server_url.rstrip("/")
        self.model_name = model_name or self._extract_model_name()
        self.timeout = timeout
        self.debug = debug
        self.target_latency_ms = target_latency_ms
        self.measure_ttft = measure_ttft

    def _extract_model_name(self) -> str:
        """Extract model name from server or use URL as fallback."""
        import requests

        try:
            response = requests.get(f"{self.server_url}/v1/models", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if "data" in data and len(data["data"]) > 0:
                    return data["data"][0].get("id", "unknown")
        except Exception:
            pass

        return f"server_{self.server_url.replace('http://', '').replace(':', '_')}"

    def run_latency_benchmark(
        self,
        test_file: Optional[Path] = None,
        warmup_runs: int = 2,
        iterations: int = 1,
        categories: Optional[list] = None,
    ) -> bool:
        """
        Run latency benchmark suite.

        Args:
            test_file: Path to latency benchmark suite JSON
            warmup_runs: Number of warmup iterations
            iterations: Number of test iterations for averaging
            categories: Filter to specific test categories (None = all)

        Returns:
            True if benchmark completed successfully
        """
        print("=" * 80)
        print("LATENCY BENCHMARK")
        print("=" * 80)
        print(f"Server: {self.server_url}")
        print(f"Model: {self.model_name}")
        print(f"Target Latency: {self.target_latency_ms}ms")
        print(f"TTFT Measurement: {'Enabled' if self.measure_ttft else 'Disabled'}")
        print()

        # Create evaluator
        evaluator = LatencyEvaluator(
            server_url=self.server_url,
            timeout=self.timeout,
            debug=self.debug,
            target_latency_ms=self.target_latency_ms,
            measure_ttft=self.measure_ttft,
        )

        # Load test suite
        test_file = test_file or DEFAULT_LATENCY_SUITE
        if not test_file.exists():
            print(f"❌ Test suite not found: {test_file}")
            return False

        try:
            suite = evaluator.load_latency_suite(test_file)
            print(f"Loaded: {suite.name}")
            print(f"Tests: {len(suite.tests)}")

            # Filter categories if specified
            if categories:
                original_count = len(suite.tests)
                suite.tests = [t for t in suite.tests if t.category in categories]
                print(f"Filtered to categories {categories}: {len(suite.tests)}/{original_count} tests")

            if not suite.tests:
                print("❌ No tests to run after filtering")
                return False

            # Run benchmark
            report = evaluator.run_benchmark(
                suite=suite,
                model_name=self.model_name,
                warmup_runs=warmup_runs,
                iterations=iterations,
            )

            # Save reports
            self._save_reports(report)

            return report.target_hit_rate >= 50

        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return False

    def run_optimization_comparison(
        self,
        test_file: Optional[Path] = None,
    ) -> bool:
        """
        Run comparison of different optimization strategies.

        Compares baseline vs prefix injection vs short fields vs combined.
        """
        print("=" * 80)
        print("OPTIMIZATION COMPARISON BENCHMARK")
        print("=" * 80)

        categories_to_compare = [
            "baseline",
            "prefix_injection",
            "short_fields",
            "combined_optimizations",
        ]

        return self.run_latency_benchmark(
            test_file=test_file,
            warmup_runs=3,
            iterations=3,
            categories=categories_to_compare,
        )

    def run_quick_latency_check(self, num_tests: int = 5) -> bool:
        """
        Run a quick latency check with baseline tests only.

        Args:
            num_tests: Number of tests to run

        Returns:
            True if average latency is below target
        """
        print("=" * 80)
        print("QUICK LATENCY CHECK")
        print("=" * 80)

        evaluator = LatencyEvaluator(
            server_url=self.server_url,
            timeout=self.timeout,
            debug=self.debug,
            target_latency_ms=self.target_latency_ms,
        )

        test_file = DEFAULT_LATENCY_SUITE
        if not test_file.exists():
            print(f"❌ Test suite not found: {test_file}")
            return False

        suite = evaluator.load_latency_suite(test_file)

        # Take only baseline tests
        baseline_tests = [t for t in suite.tests if t.category == "baseline"][:num_tests]

        if not baseline_tests:
            print("❌ No baseline tests found")
            return False

        suite.tests = baseline_tests

        # Run without warmup for quick check
        report = evaluator.run_benchmark(
            suite=suite,
            model_name=self.model_name,
            warmup_runs=1,
            iterations=1,
        )

        print(f"\n📊 Quick Check Results:")
        print(f"   Average Latency: {report.avg_latency_ms:.1f}ms")
        print(f"   Target: {self.target_latency_ms}ms")
        print(f"   Gap: {report.avg_latency_ms - self.target_latency_ms:.1f}ms")

        if report.avg_latency_ms <= self.target_latency_ms:
            print("   ✅ PASSING - within target latency")
            return True
        else:
            speedup_needed = report.avg_latency_ms / self.target_latency_ms
            print(f"   ❌ FAILING - need {speedup_needed:.1f}x speedup")
            return False

    def _save_reports(self, report) -> None:
        """Save benchmark reports to files."""
        from datetime import datetime
        import json

        # Create reports directory
        reports_dir = Path("reports/latency")
        reports_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = self.model_name.replace("/", "_").replace(":", "_")

        # Save JSON
        json_path = reports_dir / f"latency_{safe_model_name}_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(report.model_dump(), f, indent=2, default=str)

        # Save Markdown
        md_path = reports_dir / f"latency_{safe_model_name}_{timestamp}.md"
        with open(md_path, "w") as f:
            f.write(report.generate_markdown())

        print(f"\n📄 Reports saved:")
        print(f"   JSON: {json_path}")
        print(f"   Markdown: {md_path}")
