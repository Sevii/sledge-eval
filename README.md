




# Sledge Eval

Evaluation framework for testing small language models' ability to interpret voice commands and convert them into structured tool calls.

## Features

- JSON-based test definition format
- Pydantic models for type-safe test cases
- Integration with LangChain for model evaluation
- **Native support for Ministral (3B/8B) tool calling with llama.cpp**
- Extensible framework for custom evaluations

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

### Install dependencies

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install project dependencies
uv pip install -e ".[dev]"
```

## Ministral Tool Calling Setup

Ministral (3B and 8B) models are optimized for "edge" function calling and work great with llama.cpp. This framework provides native support for evaluating Ministral models on voice command interpretation tasks.

### Prerequisites

1. **Download a Ministral GGUF model:**
   - Ministral-8B-Instruct-2410 (recommended for best accuracy)
   - Ministral-3B-Instruct-2410 (faster, good for edge devices)
   - Available on HuggingFace in GGUF format (e.g., `Ministral-8B-Instruct-2410.Q4_K_M.gguf`)

2. **Install llama-cpp-python:**
   ```bash
   # CPU only
   pip install llama-cpp-python

   # With GPU support (CUDA)
   CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python

   # With GPU support (Metal - macOS)
   CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
   ```

### Quick Start with Ministral

```python
from sledge_eval import MinistralEvaluator, VoiceCommandTest, ToolCall

# Initialize the evaluator
evaluator = MinistralEvaluator(
    model_path="path/to/Ministral-8B-Instruct-2410.Q4_K_M.gguf",
    n_ctx=4096,          # Context window size
    n_gpu_layers=-1,     # -1 = use GPU for all layers, 0 = CPU only
)

# Create a test case
test = VoiceCommandTest(
    id="test_001",
    voice_command="Turn on the living room lights",
    expected_tool_calls=[
        ToolCall(name="control_lights", arguments={"room": "living room", "action": "turn_on"})
    ],
)

# Evaluate
result = evaluator.evaluate_test(test)
print(f"Test {'PASSED' if result.passed else 'FAILED'}")
print(f"Predicted: {result.predicted_tool_calls}")
```

### Ministral Configuration Options

**Key Parameters:**
- `model_path`: Path to your Ministral GGUF file (required)
- `n_ctx`: Context window size (default: 4096, can go higher)
- `n_gpu_layers`: Number of layers to offload to GPU
  - `-1`: All layers to GPU (fastest)
  - `0`: CPU only (no GPU)
  - `20-30`: Partial GPU offload (balance speed/memory)
- `verbose`: Enable detailed llama.cpp logging (default: False)
- `available_tools`: Custom tool definitions in OpenAI format

### Example Scripts

The `examples/` directory contains several demonstrations:

1. **Basic Tool Calling Demo** (`examples/ministral_tool_calling_demo.py`)
   - Direct llama.cpp usage
   - Interactive mode
   - Based on the canonical Ministral tool calling pattern

   ```bash
   # Single query
   python examples/ministral_tool_calling_demo.py --model-path /path/to/model.gguf

   # Interactive mode
   python examples/ministral_tool_calling_demo.py --model-path /path/to/model.gguf --interactive
   ```

2. **Evaluator Demo** (`examples/ministral_evaluator_demo.py`)
   - Using the MinistralEvaluator class
   - Single test evaluation
   - Full test suite evaluation
   - Custom tool definitions

   ```bash
   # Single test
   python examples/ministral_evaluator_demo.py --model-path /path/to/model.gguf

   # Full test suite
   python examples/ministral_evaluator_demo.py --model-path /path/to/model.gguf --mode suite

   # Custom tools
   python examples/ministral_evaluator_demo.py --model-path /path/to/model.gguf --mode custom
   ```

### Troubleshooting Ministral

**Issue: Model outputs JSON instead of tool calls**
- Ministral uses the Mistral V3 (Tekken) tokenizer
- Make sure you're using an up-to-date GGUF conversion
- Update llama.cpp to the latest version

**Issue: Poor tool calling accuracy**
- Always include a system prompt (e.g., "You are a helpful assistant")
- Ensure `n_ctx` is at least 4096
- Try the 8B model instead of 3B for better accuracy

**Issue: Out of memory**
- Reduce `n_gpu_layers` for partial GPU offload
- Use a smaller quantization (Q4_K_M instead of Q6_K)
- Reduce `n_ctx` if you don't need large context

## Project Structure

```
sledge-eval/
├── src/
│   └── sledge_eval/
│       ├── __init__.py
│       ├── evaluator.py              # Core evaluation logic
│       └── ministral_evaluator.py    # Ministral-specific evaluator
├── examples/
│   ├── ministral_tool_calling_demo.py     # Basic tool calling
│   └── ministral_evaluator_demo.py        # Evaluator usage
├── tests/
│   ├── test_data/
│   │   └── example_test_suite.json   # Example test definitions
│   ├── test_evaluator.py             # Core tests
│   └── test_ministral.py             # Ministral integration tests
├── pyproject.toml                    # Project configuration
└── README.md
```

## Test Definition Format

Tests are defined in JSON files using the following structure:

```json
{
  "name": "Test Suite Name",
  "description": "Description of the test suite",
  "tests": [
    {
      "id": "test_001",
      "voice_command": "Turn on the living room lights",
      "description": "Optional description",
      "tags": ["smart_home", "lights"],
      "expected_tool_calls": [
        {
          "name": "control_lights",
          "arguments": {
            "room": "living room",
            "action": "turn_on"
          }
        }
      ]
    }
  ]
}
```

### Test Case Fields

- `id`: Unique identifier for the test
- `voice_command`: The voice command text to be interpreted
- `expected_tool_calls`: List of expected tool calls
  - `name`: Function/tool name
  - `arguments`: Dictionary of arguments to pass to the tool
- `description`: Optional description of what the test validates
- `tags`: List of tags for categorizing tests

## Running Tests

```bash
# Run all tests (except Ministral integration tests)
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_evaluator.py

# Run tests with specific marker
pytest -m unit
```

### Running Ministral Integration Tests

The Ministral integration tests require a GGUF model file. Set the `MINISTRAL_MODEL_PATH` environment variable to enable these tests:

```bash
# Set model path and run Ministral tests
export MINISTRAL_MODEL_PATH="/path/to/Ministral-8B-Instruct-2410.Q4_K_M.gguf"
pytest tests/test_ministral.py -v

# Run only integration tests
pytest -m integration

# Skip integration tests
pytest -m "not integration"
```

## Code Formatting

This project uses [Black](https://black.readthedocs.io/) for code formatting.

```bash
# Format all code
black src tests

# Check formatting without making changes
black --check src tests
```

## Usage

### With Ministral (Recommended)

```python
from pathlib import Path
from sledge_eval import MinistralEvaluator

# Initialize Ministral evaluator
evaluator = MinistralEvaluator(
    model_path="path/to/Ministral-8B-Instruct-2410.Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=-1,  # Use GPU
)

# Load test suite
test_suite = evaluator.load_test_suite(Path("tests/test_data/example_test_suite.json"))

# Run evaluation
results = evaluator.evaluate_suite(test_suite)

# Analyze results
for result in results:
    status = "PASS" if result.passed else "FAIL"
    print(f"Test {result.test_id}: {status}")
    if not result.passed:
        print(f"  Expected: {result.expected_tool_calls}")
        print(f"  Predicted: {result.predicted_tool_calls}")
```

### With Custom Model

```python
from pathlib import Path
from sledge_eval.evaluator import Evaluator

# Initialize evaluator with your model client
evaluator = Evaluator(model_client=your_model)

# Load test suite
test_suite = evaluator.load_test_suite(Path("tests/test_data/example_test_suite.json"))

# Run evaluation (requires implementing evaluate_test method)
results = evaluator.evaluate_suite(test_suite)

# Analyze results
for result in results:
    print(f"Test {result.test_id}: {'PASS' if result.passed else 'FAIL'}")
```

## Development

### Adding New Tests

1. Create or modify JSON test files in `tests/test_data/`
2. Follow the test definition format described above
3. Run tests to validate the format

### Implementing Model Integration

The `Evaluator.evaluate_test()` method needs to be implemented with your specific model inference logic:

```python
def evaluate_test(self, test: VoiceCommandTest) -> EvaluationResult:
    # 1. Pass voice_command to your model
    # 2. Parse model output into ToolCall objects
    # 3. Compare with expected_tool_calls
    # 4. Return EvaluationResult
    pass
```


Qwen/Qwen3-VL-4B-Instruct-GGUF
mistralai/Ministral-3-14B-Reasoning-2512-GGUF
mistralai/Ministral-3-3B-Reasoning-2512-GGUF



## License

See LICENSE file for details.
