"""Sledge Eval: Evaluation framework for voice command to tool call interpretation."""

from .evaluator import (
    EvaluationReport,
    EvaluationResult,
    Evaluator,
    TestSuite,
    ToolCall,
    VoiceCommandTest,
    TextEvaluationTest,
    TextEvaluationSuite,
    TextEvaluationResult,
)
# Optional imports for evaluators with dependencies
try:
    from .ministral_evaluator import MinistralEvaluator
except ImportError:
    MinistralEvaluator = None

try:
    from .gemini_evaluator import GeminiEvaluator, GeminiTextEvaluator
except ImportError:
    GeminiEvaluator = None
    GeminiTextEvaluator = None

try:
    from .gemini_anki_evaluator import GeminiAnkiEvaluator
except ImportError:
    GeminiAnkiEvaluator = None

from .server_evaluator import ServerEvaluator
from .anki_evaluator import AnkiLargeToolSetEvaluator
from .text_evaluator import TextEvaluator
from .text_server_evaluator import TextServerEvaluator
from .hardware_detector import HardwareInfo, HardwareDetector

__version__ = "0.1.0"

# Build __all__ dynamically based on what was successfully imported
__all__ = [
    "AnkiLargeToolSetEvaluator",
    "EvaluationReport",
    "EvaluationResult", 
    "Evaluator",
    "HardwareDetector",
    "HardwareInfo",
    "ServerEvaluator",
    "TestSuite",
    "TextEvaluationTest",
    "TextEvaluationSuite", 
    "TextEvaluationResult",
    "TextEvaluator",
    "TextServerEvaluator",
    "ToolCall",
    "VoiceCommandTest",
]

# Add MinistralEvaluator to __all__ only if it was successfully imported
if MinistralEvaluator is not None:
    __all__.append("MinistralEvaluator")

# Add GeminiEvaluator to __all__ only if it was successfully imported
if GeminiEvaluator is not None:
    __all__.append("GeminiEvaluator")

# Add GeminiTextEvaluator to __all__ only if it was successfully imported
if GeminiTextEvaluator is not None:
    __all__.append("GeminiTextEvaluator")

# Add GeminiAnkiEvaluator to __all__ only if it was successfully imported
if GeminiAnkiEvaluator is not None:
    __all__.append("GeminiAnkiEvaluator")
