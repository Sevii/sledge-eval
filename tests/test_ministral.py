"""Tests for Ministral tool calling functionality.

Note: These tests require a Ministral GGUF model to be available.
Set the MINISTRAL_MODEL_PATH environment variable to run these tests.
"""

import os
from pathlib import Path

import pytest

from sledge_eval.ministral_evaluator import MinistralEvaluator
from sledge_eval.evaluator import ToolCall, VoiceCommandTest

# Skip all tests in this module if model path is not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("MINISTRAL_MODEL_PATH"),
    reason="MINISTRAL_MODEL_PATH environment variable not set",
)


@pytest.fixture
def model_path():
    """Get the model path from environment variable."""
    return os.environ.get("MINISTRAL_MODEL_PATH")


@pytest.fixture
def ministral_evaluator(model_path):
    """Create a MinistralEvaluator instance."""
    return MinistralEvaluator(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=-1,  # Use GPU if available
        verbose=False,
    )


@pytest.mark.integration
def test_ministral_evaluator_initialization(model_path):
    """Test that MinistralEvaluator can be initialized."""
    evaluator = MinistralEvaluator(model_path=model_path)
    assert evaluator.llm is not None
    assert len(evaluator.available_tools) > 0


@pytest.mark.integration
def test_simple_light_control(ministral_evaluator):
    """Test a simple light control command."""
    test = VoiceCommandTest(
        id="test_light_001",
        voice_command="Turn on the living room lights",
        expected_tool_calls=[
            ToolCall(name="control_lights", arguments={"room": "living room", "action": "turn_on"})
        ],
    )

    result = ministral_evaluator.evaluate_test(test)

    assert result.test_id == "test_light_001"
    assert result.error is None
    assert len(result.predicted_tool_calls) > 0

    # Check that the model predicted a control_lights call
    predicted_names = [call.name for call in result.predicted_tool_calls]
    assert "control_lights" in predicted_names


@pytest.mark.integration
def test_thermostat_control(ministral_evaluator):
    """Test a thermostat control command."""
    test = VoiceCommandTest(
        id="test_thermo_001",
        voice_command="Set the thermostat to 72 degrees",
        expected_tool_calls=[
            ToolCall(name="set_temperature", arguments={"temperature": 72, "unit": "fahrenheit"})
        ],
    )

    result = ministral_evaluator.evaluate_test(test)

    assert result.test_id == "test_thermo_001"
    assert result.error is None
    assert len(result.predicted_tool_calls) > 0

    # Check that the model predicted a set_temperature call
    predicted_names = [call.name for call in result.predicted_tool_calls]
    assert "set_temperature" in predicted_names


@pytest.mark.integration
def test_weather_query(ministral_evaluator):
    """Test a weather query command."""
    test = VoiceCommandTest(
        id="test_weather_001",
        voice_command="What's the weather like today?",
        expected_tool_calls=[
            ToolCall(name="get_weather", arguments={"timeframe": "today"})
        ],
    )

    result = ministral_evaluator.evaluate_test(test)

    assert result.test_id == "test_weather_001"
    assert result.error is None


@pytest.mark.integration
def test_multi_action_command(ministral_evaluator):
    """Test a command that requires multiple tool calls."""
    test = VoiceCommandTest(
        id="test_multi_001",
        voice_command="Play my workout playlist and turn up the volume",
        expected_tool_calls=[
            ToolCall(name="play_music", arguments={"playlist": "workout"}),
            ToolCall(name="adjust_volume", arguments={"action": "increase"}),
        ],
    )

    result = ministral_evaluator.evaluate_test(test)

    assert result.test_id == "test_multi_001"
    assert result.error is None
    # Note: Not all models may handle multi-action commands perfectly


@pytest.mark.integration
def test_evaluate_full_suite(ministral_evaluator):
    """Test evaluating a full test suite."""
    test_file = Path(__file__).parent / "test_data" / "example_test_suite.json"
    test_suite = ministral_evaluator.load_test_suite(test_file)

    results = ministral_evaluator.evaluate_suite(test_suite)

    assert len(results) == len(test_suite.tests)
    assert all(result.test_id is not None for result in results)

    # Print results for debugging
    passed = sum(1 for r in results if r.passed)
    print(f"\nTest Results: {passed}/{len(results)} passed")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"  {result.test_id}: {status}")
        if not result.passed:
            print(f"    Expected: {result.expected_tool_calls}")
            print(f"    Predicted: {result.predicted_tool_calls}")


def test_tool_call_comparison():
    """Test the tool call comparison logic."""
    evaluator = MinistralEvaluator.__new__(MinistralEvaluator)

    # Exact match
    pred = [ToolCall(name="test_func", arguments={"key": "value"})]
    exp = [ToolCall(name="test_func", arguments={"key": "value"})]
    assert evaluator._compare_tool_calls(pred, exp) is True

    # Different names
    pred = [ToolCall(name="func1", arguments={"key": "value"})]
    exp = [ToolCall(name="func2", arguments={"key": "value"})]
    assert evaluator._compare_tool_calls(pred, exp) is False

    # Different argument values
    pred = [ToolCall(name="test_func", arguments={"key": "value1"})]
    exp = [ToolCall(name="test_func", arguments={"key": "value2"})]
    assert evaluator._compare_tool_calls(pred, exp) is False

    # Missing argument
    pred = [ToolCall(name="test_func", arguments={})]
    exp = [ToolCall(name="test_func", arguments={"key": "value"})]
    assert evaluator._compare_tool_calls(pred, exp) is False

    # Extra arguments in prediction (should still pass)
    pred = [ToolCall(name="test_func", arguments={"key": "value", "extra": "data"})]
    exp = [ToolCall(name="test_func", arguments={"key": "value"})]
    assert evaluator._compare_tool_calls(pred, exp) is True
