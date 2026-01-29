# Sledge Eval

Voice command to tool call evaluation framework for LLMs. Tests models on their ability to convert natural language commands into structured tool/function calls.

## Quick Start

```bash
# Run all evaluations against local llama-server
python eval_server.py --port 8080

# Run evaluations via OpenRouter API
python eval_openrouter.py --model anthropic/claude-3-haiku

# Run evaluations via Google Gemini API
python eval_gemini.py --model gemini-2.5-flash-lite

# Update the leaderboard from reports
python generate_leaderboard.py

# Run tests
pytest tests/
```

## Entry Points

### eval_server.py - Local llama-server

Evaluate models running on a local llama-server instance.

```bash
python eval_server.py --port 8080                    # Default: run all tests
python eval_server.py --server-url http://localhost:8080 --mode suite
python eval_server.py --port 8080 --mode anki --debug
python eval_server.py --port 8080 --mode text --test-suite tests/test_data/comprehensive_text_suite.json
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--server-url` | - | Full URL of llama-server instance |
| `--port` | 8080 | Port number (builds URL as `http://localhost:PORT`) |
| `--mode` | all | Evaluation mode: `single`, `suite`, `custom`, `all`, `anki`, `text` |
| `--test-suite` | - | Path to custom test suite JSON |
| `--timeout` | 120 | Request timeout in seconds |
| `--debug` | false | Enable verbose logging |
| `--model-name` | - | Override model name for reports (auto-detects if not set) |

### eval_openrouter.py - OpenRouter API

Evaluate models via the OpenRouter API.

```bash
python eval_openrouter.py --model anthropic/claude-3-haiku
python eval_openrouter.py --model openai/gpt-4o --mode all --debug
python eval_openrouter.py --model meta-llama/llama-3-70b-instruct --mode anki
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | **required** | OpenRouter model ID (e.g., `anthropic/claude-3-haiku`) |
| `--mode` | all | Evaluation mode: `single`, `suite`, `custom`, `all`, `anki`, `text` |
| `--test-suite` | - | Path to custom test suite JSON |
| `--api-key` | - | OpenRouter API key (or use `OPENROUTER_API_KEY` env var) |
| `--timeout` | 120 | Request timeout in seconds |
| `--debug` | false | Enable verbose logging |
| `--site-url` | - | Site URL for OpenRouter ranking |
| `--app-name` | sledge-eval | App name for OpenRouter ranking |

### eval_gemini.py - Google Gemini API

Evaluate Google Gemini models.

```bash
python eval_gemini.py                                # Default model: gemini-2.5-flash-lite
python eval_gemini.py --model gemini-2.5-flash-lite
python eval_gemini.py --mode suite --test-suite tests/test_data/example_test_suite.json
python eval_gemini.py --debug
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | gemini-2.5-flash-lite | Gemini model to use |
| `--mode` | all | Evaluation mode: `single`, `suite`, `custom`, `all`, `anki`, `text` |
| `--test-suite` | - | Path to custom test suite JSON |
| `--api-key` | - | Google API key (or use `GEMINI_API_KEY` env var) |
| `--debug` | false | Enable verbose logging |

### eval_latency.py - Latency Benchmarking

Measure and optimize LLM inference latency. Target: 100ms for 4B parameter models.

```bash
python eval_latency.py --port 8080                   # Full benchmark suite
python eval_latency.py --port 8080 --mode quick      # Quick latency check
python eval_latency.py --port 8080 --mode compare    # Compare optimization strategies
python eval_latency.py --port 8080 --target 150      # Custom target latency (ms)
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--server-url` | - | Full URL of llama-server instance |
| `--port` | 8080 | Port number |
| `--mode` | full | Benchmark mode: `full`, `quick`, `compare` |
| `--target` | 100 | Target latency in milliseconds |
| `--warmup` | 2 | Number of warmup runs |
| `--iterations` | 1 | Number of test iterations for averaging |
| `--test-suite` | - | Path to custom latency test suite JSON |
| `--category` | - | Filter to specific test categories (can repeat) |
| `--measure-ttft` | false | Measure Time to First Token (requires streaming) |
| `--timeout` | 120 | Request timeout in seconds |
| `--debug` | false | Enable verbose logging |
| `--model-name` | - | Override model name for reports |

### generate_leaderboard.py - Leaderboard Generation

Generate a markdown leaderboard from evaluation reports.

```bash
python generate_leaderboard.py                       # Default: reports/ -> LEADERBOARD.md
python generate_leaderboard.py --reports-dir reports --output LEADERBOARD.md
python generate_leaderboard.py --print               # Also print to stdout
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--reports-dir` | reports | Directory containing model report subdirectories |
| `--output`, `-o` | LEADERBOARD.md | Output markdown file |
| `--print`, `-p` | false | Print leaderboard to stdout |

## Evaluation Modes

| Mode | Description |
|------|-------------|
| `single` | Run a single test case |
| `suite` | Run a test suite from JSON file |
| `custom` | Run with custom tool definitions |
| `all` | Run comprehensive evaluation (default) |
| `anki` | Large toolset testing (50+ tools) |
| `text` | Text-based QA evaluation |

## Project Structure

```
sledge-eval/
├── eval_server.py           # Local llama-server entry point
├── eval_openrouter.py       # OpenRouter API entry point
├── eval_gemini.py           # Google Gemini API entry point
├── eval_latency.py          # Latency benchmarking entry point
├── generate_leaderboard.py  # Leaderboard generation
├── src/sledge_eval/         # Main package
│   ├── cli/                 # CLI runners
│   ├── evaluator.py         # Core evaluation logic
│   ├── config.py            # Configuration
│   ├── tools/               # Tool definitions
│   ├── utils/               # Utilities
│   └── reporting/           # Report generation
├── tests/                   # Test suite
│   └── test_data/           # Test data files
│       ├── example_test_suite.json
│       ├── anki_large_toolset_suite.json
│       ├── comprehensive_text_suite.json
│       ├── latency_benchmark_suite.json
│       ├── letter_counting_suite.json
│       └── theory_of_mind_suite.json
├── reports/                 # Generated evaluation reports
├── LEADERBOARD.md           # Generated leaderboard
└── .env                     # API keys (not in git)
```

## API Keys

Create a `.env` file in the project root:

```bash
# OpenRouter API
OPENROUTER_API_KEY=sk-or-v1-xxx

# Google Gemini API
GEMINI_API_KEY=xxx
```

Or pass keys directly via command line arguments (`--api-key`).

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_evaluator.py
```
