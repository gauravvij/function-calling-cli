"""
FC-Eval: Function Calling Evaluation Tool for OpenRouter LLMs

A CLI tool for evaluating Large Language Models' function-calling capabilities
inspired by the Berkeley Function Calling Leaderboard (BFCL) v4 methodology.

Features:
- Best of N trials support for reliable evaluation
- Reliability metrics showing consistency across trials
- Parallel and sequential execution modes
- Comprehensive reporting with JSON and TXT outputs
- 30 unique test cases covering single_turn, multi_turn, and agentic scenarios
"""

__version__ = "1.0.0"
__author__ = "NEO - A fully autonomous AI Engineer"
__url__ = "https://heyneo.so"

from .client import OpenRouterClient
from .evaluator import EvaluationExecutor
from .reporter import ReportGenerator
from .test_suite import BFCLTestSuite, TestCase
from .validator import ASTValidator

__all__ = [
    "OpenRouterClient",
    "EvaluationExecutor",
    "ReportGenerator",
    "BFCLTestSuite",
    "TestCase",
    "ASTValidator",
]
