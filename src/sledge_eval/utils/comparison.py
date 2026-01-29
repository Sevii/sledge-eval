"""Comparison utilities for evaluation results."""

from typing import Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..evaluator import ToolCall


def _normalize_keys(obj: Any) -> Any:
    """Recursively normalize dictionary keys to lowercase."""
    if isinstance(obj, dict):
        return {k.lower(): _normalize_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_normalize_keys(item) for item in obj]
    return obj


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

        # Normalize dictionary keys to lowercase for case-insensitive comparison
        pred_args = _normalize_keys(pred.arguments)
        exp_args = _normalize_keys(exp.arguments)

        if strict:
            # Strict mode: arguments must match exactly
            if pred_args != exp_args:
                return False
        else:
            # Flexible mode: all expected arguments must be present with correct values
            for key, value in exp_args.items():
                if key not in pred_args:
                    return False
                if pred_args[key] != value:
                    return False

    return True
