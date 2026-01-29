#!/usr/bin/env python3
"""
Sledge Eval Latency Benchmark CLI

Experimental tool for measuring and optimizing LLM inference latency.
Target: 100ms for 4B parameter models.

Usage:
    python eval_latency.py --port 8080
    python eval_latency.py --port 8080 --mode quick
    python eval_latency.py --port 8080 --mode compare
    python eval_latency.py --port 8080 --target 150

Benchmark Modes:
    full     - Run complete latency benchmark suite
    quick    - Quick latency check (5 tests, no warmup)
    compare  - Compare optimization strategies (baseline vs optimized)

Optimization Categories Tested:
    baseline              - Standard prompts, normal field names
    prefix_injection      - Model starts mid-JSON response
    short_fields          - Minimal token count tool definitions
    combined_optimizations - Both prefix injection and short fields
    speculative_friendly  - Predictable patterns for speculative decoding
"""

import argparse
from pathlib import Path

from src.sledge_eval.cli.latency_runner import LatencyBenchmarkRunner


def main():
    """Main entry point for latency benchmark CLI."""
    parser = argparse.ArgumentParser(
        description="Sledge Eval - Latency Benchmark for LLM Inference Optimization",
        prog="python eval_latency.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Optimization Recommendations:
  If latency > 350ms:
    - Switch to TensorRT-LLM or SGLang
    - Enable INT4 quantization (AWQ/GPTQ)

  If latency 200-350ms:
    - Enable speculative decoding
    - Use prefix injection
    - Shorten field names

  If latency 100-200ms:
    - Fine-tune batch size
    - Optimize KV cache

  Target: 100ms = 20ms TTFT + 80ms generation (30 tokens @ 375 t/s)
        """
    )

    parser.add_argument(
        "--server-url",
        type=str,
        help="URL of the llama-server instance",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port of the llama-server instance (default: 8080)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "quick", "compare"],
        default="full",
        help="Benchmark mode: full suite, quick check, or optimization comparison (default: full)",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=100.0,
        help="Target latency in milliseconds (default: 100)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup runs before measurement (default: 2)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of test iterations for averaging (default: 1)",
    )
    parser.add_argument(
        "--test-suite",
        type=str,
        help="Path to custom latency test suite JSON",
    )
    parser.add_argument(
        "--category",
        type=str,
        action="append",
        help="Filter to specific test categories (can be used multiple times)",
    )
    parser.add_argument(
        "--measure-ttft",
        action="store_true",
        help="Measure Time to First Token (requires streaming support)",
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
        help="Enable debug mode with verbose logging",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Override model name for reports",
    )

    args = parser.parse_args()

    # Determine server URL
    server_url = args.server_url or f"http://localhost:{args.port}"

    print(f"🚀 Sledge Eval Latency Benchmark")
    print(f"Server: {server_url}")
    print(f"Mode: {args.mode}")
    print(f"Target: {args.target}ms")
    if args.debug:
        print(f"Debug: ENABLED")
    print()

    try:
        # Create runner
        runner = LatencyBenchmarkRunner(
            server_url=server_url,
            model_name=args.model_name,
            timeout=args.timeout,
            debug=args.debug,
            target_latency_ms=args.target,
            measure_ttft=args.measure_ttft,
        )

        # Parse test file path if provided
        test_file = Path(args.test_suite) if args.test_suite else None

        # Run appropriate mode
        success = False

        if args.mode == "quick":
            success = runner.run_quick_latency_check()

        elif args.mode == "compare":
            success = runner.run_optimization_comparison(test_file)

        elif args.mode == "full":
            success = runner.run_latency_benchmark(
                test_file=test_file,
                warmup_runs=args.warmup,
                iterations=args.iterations,
                categories=args.category,
            )

        # Print final status
        print()
        if success:
            print("🎯 Benchmark completed - meeting target latency!")
            return 0
        else:
            print("⚠️  Benchmark completed - not meeting target latency")
            print("   Run with --mode compare to see optimization impact")
            return 1

    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nError during benchmark: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
