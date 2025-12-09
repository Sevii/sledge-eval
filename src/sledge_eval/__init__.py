"""Sledge Eval: Evaluation framework for voice command to tool call interpretation."""

from .evaluator import (
    EvaluationResult,
    Evaluator,
    TestSuite,
    ToolCall,
    VoiceCommandTest,
)
from .ministral_evaluator import MinistralEvaluator
from .server_evaluator import ServerEvaluator

__version__ = "0.1.0"

__all__ = [
    "EvaluationResult",
    "Evaluator",
    "MinistralEvaluator",
    "ServerEvaluator",
    "TestSuite",
    "ToolCall",
    "VoiceCommandTest",
]
