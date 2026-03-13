"""
Command-line interface for FC-Eval.
"""

import os
import sys
import json
import time
import logging
import argparse
from typing import List, Optional

from .client import OpenRouterClient, OllamaClient
from .evaluator import EvaluationExecutor
from .reporter import ReportGenerator
from .test_suite import BFCLTestSuite

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('evaluation.log')
    ]
)
logger = logging.getLogger(__name__)


# Default models to evaluate
DEFAULT_MODELS = [
    "x-ai/grok-4.1-fast",
    "google/gemini-3-flash-preview",
    "qwen/qwen3.5-27b",
    "x-ai/grok-4.20-beta",
    "moonshotai/kimi-k2.5",
    "qwen/qwen3.5-122b-a10b",
    "minimax/minimax-m2.5",
    "google/gemini-3.1-flash-lite-preview",
    "qwen/qwen3.5-35b-a3b",
    "qwen/qwen3.5-flash-02-23",
    "anthropic/claude-haiku-4.5"
]


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="FC-Eval: Evaluate LLM function calling capabilities with Best of N trials (OpenRouter or Ollama)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # OpenRouter (default)
  fc-eval --provider openrouter --models openai/gpt-4o anthropic/claude-3.5-sonnet
  
  # Ollama (local)
  fc-eval --provider ollama --models llama3.2 mistral
  
  # Other options
  fc-eval --mode parallel --trials 5 --max-workers 10
  fc-eval --category single_turn --output-dir ./my_results

For more information, visit: https://github.com/gauravvij/function-calling-cli
        """
    )
    parser.add_argument(
        "--provider",
        choices=["openrouter", "ollama"],
        default="openrouter",
        help="API provider to use for evaluation (default: openrouter)"
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama API base URL (default: http://localhost:11434)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="List of models to evaluate (default: predefined list for OpenRouter, must specify for Ollama)"
    )
    parser.add_argument(
        "--mode",
        choices=["sequential", "parallel"],
        default="parallel",
        help="Execution mode: sequential or parallel (default: parallel)"
    )
    parser.add_argument(
        "--category",
        choices=["single_turn", "multi_turn", "agentic", "all"],
        default="all",
        help="Filter tests by category (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        default="./results",
        help="Directory for output reports (default: ./results)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum parallel workers (default: 5)"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of trials per test case (default: 3, Best of N logic)"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the evaluation script."""
    args = parse_arguments()
    
    # Initialize appropriate client based on provider
    try:
        if args.provider == "ollama":
            client = OllamaClient(base_url=args.ollama_url)
            logger.info(f"Ollama client initialized successfully (URL: {args.ollama_url})")
            # Validate that models are specified for Ollama
            if not args.models:
                logger.error("Ollama provider requires --models to be specified (e.g., --models llama3.2 mistral)")
                sys.exit(1)
        else:
            client = OpenRouterClient(api_key=args.api_key)
            logger.info("OpenRouter client initialized successfully")
    except ValueError as e:
        logger.error(f"Failed to initialize {args.provider} client: {e}")
        sys.exit(1)
    
    test_suite = BFCLTestSuite()
    
    if args.category == "all":
        test_cases = test_suite.get_all_tests()
    else:
        test_cases = test_suite.get_tests_by_category(args.category)
    
    logger.info(f"Loaded {len(test_cases)} test cases (category: {args.category})")
    
    models = args.models if args.models else DEFAULT_MODELS
    logger.info(f"Models to evaluate: {models}")
    logger.info(f"Trials per test: {args.trials} (Best of N: pass if at least one trial succeeds)")
    
    executor = EvaluationExecutor(client, max_workers=args.max_workers, trials=args.trials)
    
    start_time = time.time()
    
    if args.mode == "sequential":
        logger.info("Running evaluation in SEQUENTIAL mode")
        results = executor.run_sequential(models, test_cases)
    else:
        logger.info("Running evaluation in PARALLEL mode")
        results = executor.run_parallel(models, test_cases, max_workers=args.max_workers)
    
    elapsed = time.time() - start_time
    logger.info(f"Evaluation completed in {elapsed:.1f} seconds")
    
    # Generate reports
    report_gen = ReportGenerator(results)
    text_path, json_path = report_gen.save_reports(args.output_dir)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Total API calls: {len(models) * len(test_cases) * args.trials}")
    print(f"Models evaluated: {len(models)}")
    print(f"Tests per model: {len(test_cases)}")
    print(f"Trials per test: {args.trials}")
    print(f"Execution time: {elapsed:.1f} seconds")
    print(f"\nReports saved:")
    print(f"  Text: {text_path}")
    print(f"  JSON: {json_path}")
    print("=" * 80)
    
    return 0
