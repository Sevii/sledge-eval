# Sledge Eval

Evaluation framework for testing small language models' ability to interpret voice commands and convert them into structured tool calls.

## Features

- JSON-based test definition format
- Pydantic models for type-safe test cases
- Integration with LangChain for model evaluation
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

## Project Structure

```
sledge-eval/
├── src/
│   └── sledge_eval/
│       ├── __init__.py
│       └── evaluator.py       # Core evaluation logic
├── tests/
│   ├── test_data/
│   │   └── example_test_suite.json  # Example test definitions
│   └── test_evaluator.py     # Test cases
├── pyproject.toml             # Project configuration
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
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_evaluator.py

# Run tests with specific marker
pytest -m unit
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

```python
from pathlib import Path
from sledge_eval.evaluator import Evaluator

# Initialize evaluator with your model client
evaluator = Evaluator(model_client=your_model)

# Load test suite
test_suite = evaluator.load_test_suite(Path("tests/test_data/example_test_suite.json"))

# Run evaluation
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

## License

See LICENSE file for details.
