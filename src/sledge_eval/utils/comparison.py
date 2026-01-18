"""Comparison utilities for evaluation results."""

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..evaluator import ToolCall


def compare_tool_calls(
    predicted: List["ToolCall"],
    expected: List["ToolCall"],
    strict: bool = False,
) -> bool:
    """
    Compare predicted and expected tool calls.

    Args:
        predicted: List of predicted tool calls
        expected: List of expected tool calls
        strict: If True, requires exact match of all arguments.
                If False, only requires expected arguments to be present.

    Returns:
        True if they match, False otherwise
    """
    if len(predicted) != len(expected):
        return False

    for pred, exp in zip(predicted, expected):
        # Check function names match
        if pred.name != exp.name:
            return False

        if strict:
            # Strict mode: arguments must match exactly
            if pred.arguments != exp.arguments:
                return False
        else:
            # Flexible mode: all expected arguments must be present with correct values
            for key, value in exp.arguments.items():
                if key not in pred.arguments:
                    return False
                if pred.arguments[key] != value:
                    return False

    return True
