"""Tests for the evaluator module."""

from pathlib import Path

import pytest

from sledge_eval.evaluator import (
    Evaluator,
    TestSuite,
    ToolCall,
    VoiceCommandTest,
)


def test_load_test_suite():
    """Test loading a test suite from JSON."""
    test_file = Path(__file__).parent / "test_data" / "example_test_suite.json"
    evaluator = Evaluator(model_client=None)
    test_suite = evaluator.load_test_suite(test_file)

    assert isinstance(test_suite, TestSuite)
    assert test_suite.name == "Basic Voice Commands"
    assert len(test_suite.tests) == 4


def test_voice_command_test_model():
    """Test VoiceCommandTest pydantic model."""
    test = VoiceCommandTest(
        id="test_001",
        voice_command="Turn on the lights",
        expected_tool_calls=[
            ToolCall(name="control_lights", arguments={"action": "turn_on"})
        ],
        description="Test turning on lights",
        tags=["lights", "basic"],
    )

    assert test.id == "test_001"
    assert test.voice_command == "Turn on the lights"
    assert len(test.expected_tool_calls) == 1
    assert test.expected_tool_calls[0].name == "control_lights"


def test_tool_call_model():
    """Test ToolCall pydantic model."""
    tool_call = ToolCall(name="test_function", arguments={"param1": "value1"})

    assert tool_call.name == "test_function"
    assert tool_call.arguments == {"param1": "value1"}


def test_tool_call_without_arguments():
    """Test ToolCall with no arguments."""
    tool_call = ToolCall(name="test_function")

    assert tool_call.name == "test_function"
    assert tool_call.arguments == {}
