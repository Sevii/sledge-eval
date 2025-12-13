




# Sledge Eval

Evaluation framework for testing small language models' ability to interpret voice commands and convert them into structured tool calls.

## Features

- JSON-based test definition format
- Pydantic models for type-safe test cases
- Calls the llama-server completions API

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

### Install dependencies

```bash
# Install uv if you haven't already
https://github.com/astral-sh/uv

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install project dependencies
uv pip install -e ".[dev]"
```

### Running it 

` ./run_eval.sh`

`./run_eval.sh mistralai/Ministral-3-3B-Reasoning-2512-GGUF`

- Boots a llama.cpp server
- Runs tests against it 
- Cleans up server
- Reports results

`uv run eval_server.py`
Runs the tests against a llama.cpp server you setup to run at localhost:8080


### Models I've tested
- Qwen/Qwen3-VL-4B-Instruct-GGUF
- mistralai/Ministral-3-3B-Reasoning-2512-GGUF
- mistralai/Ministral-3-14B-Reasoning-2512-GGUF

- Qwen/Qwen3-VL-8B-Instruct-GGUF
- Qwen/Qwen3-VL-8B-Thinking-GGUF

#TODO 

- Add test cases written by a human
- Experiment with how different tool suites effect performance.




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




## License


MIT

See LICENSE file for details.
