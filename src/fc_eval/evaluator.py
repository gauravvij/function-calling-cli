"""
Evaluation executor for running test cases against models.
"""

import time
import logging
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from .client import OpenRouterClient
from .test_suite import TestCase
from .validator import ASTValidator

logger = logging.getLogger(__name__)


@dataclass
class TrialResult:
    """Represents the result of a single trial."""
    trial_number: int
    passed: bool
    response: Optional[str]
    parsed_calls: List[dict]
    error: Optional[str] = None
    latency_ms: float = 0.0
    tokens_used: int = 0


@dataclass
class TestResult:
    """Represents the result of a single test case execution with multiple trials."""
    test_id: str
    category: str
    subcategory: str
    model: str
    passed: bool  # Best of N: True if at least one trial passed
    reliability: float  # Percentage of trials that passed
    trials: List[TrialResult]  # All trial results
    expected_calls: List[dict]
    error: Optional[str] = None
    avg_latency_ms: float = 0.0
    total_tokens: int = 0


class EvaluationExecutor:
    """Executes test cases against models with sequential or parallel support."""
    
    def __init__(self, client: OpenRouterClient, max_workers: int = 5, trials: int = 3):
        self.client = client
        self.max_workers = max_workers
        self.trials = trials
        self.validator = ASTValidator()
    
    def run_single_trial(self, model: str, test_case: TestCase, trial_number: int) -> TrialResult:
        """Run a single trial for a test case."""
        start_time = time.time()
        
        try:
            api_messages = []
            for msg in test_case.messages:
                if msg.get("role") in ["user", "assistant", "system"]:
                    api_msg = {"role": msg["role"], "content": msg.get("content", "")}
                    api_messages.append(api_msg)
            
            response = self.client.chat_completion(
                model=model,
                messages=api_messages,
                tools=test_case.tools,
                temperature=0.0
            )
            
            latency_ms = (time.time() - start_time) * 1000
            parsed_calls = self.client.extract_tool_calls(response)
            tokens_used = response.get("usage", {}).get("total_tokens", 0)
            
            raw_content = ""
            if "choices" in response and len(response["choices"]) > 0:
                raw_content = response["choices"][0].get("message", {}).get("content", "")
            
            passed, message = self.validator.evaluate_test(test_case.expected_calls, parsed_calls)
            
            return TrialResult(
                trial_number=trial_number,
                passed=passed,
                response=raw_content,
                parsed_calls=parsed_calls,
                error=None if passed else message,
                latency_ms=latency_ms,
                tokens_used=tokens_used
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Error in trial {trial_number} for test {test_case.id} with model {model}: {e}")
            return TrialResult(
                trial_number=trial_number,
                passed=False,
                response=None,
                parsed_calls=[],
                error=str(e),
                latency_ms=latency_ms,
                tokens_used=0
            )
    
    def run_single_test(self, model: str, test_case: TestCase) -> TestResult:
        """Run multiple trials for a single test case against a model."""
        trials = []
        
        logger.debug(f"Running {self.trials} trials for {model} - {test_case.id}")
        
        for trial_num in range(1, self.trials + 1):
            trial_result = self.run_single_trial(model, test_case, trial_num)
            trials.append(trial_result)
            if trial_num < self.trials:
                time.sleep(0.3)
        
        passed_trials = [t for t in trials if t.passed]
        passed = len(passed_trials) > 0
        reliability = (len(passed_trials) / len(trials)) * 100 if trials else 0.0
        
        latencies = [t.latency_ms for t in trials if t.latency_ms > 0]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        total_tokens = sum(t.tokens_used for t in trials)
        
        error = None
        if not passed:
            errors = [t.error for t in trials if t.error]
            if errors:
                error = f"Failed {len(trials) - len(passed_trials)}/{len(trials)} trials. Last error: {errors[-1]}"
        
        return TestResult(
            test_id=test_case.id,
            category=test_case.category,
            subcategory=test_case.subcategory,
            model=model,
            passed=passed,
            reliability=reliability,
            trials=trials,
            expected_calls=test_case.expected_calls,
            error=error,
            avg_latency_ms=avg_latency,
            total_tokens=total_tokens
        )
    
    def run_sequential(self, models: List[str], test_cases: List[TestCase]) -> List[TestResult]:
        """Run evaluation sequentially for all models and tests."""
        results = []
        total = len(models) * len(test_cases)
        completed = 0
        
        logger.info(f"Starting sequential evaluation: {len(models)} models x {len(test_cases)} tests x {self.trials} trials = {total * self.trials} total calls")
        
        for model in models:
            for test_case in test_cases:
                result = self.run_single_test(model, test_case)
                results.append(result)
                completed += 1
                
                status = "✓" if result.passed else "✗"
                reliability_str = f"({result.reliability:.0f}% reliable)"
                logger.info(f"[{completed}/{total}] {status} {model} - {test_case.id} {reliability_str}")
                time.sleep(0.5)
        
        return results
    
    def run_parallel(self, models: List[str], test_cases: List[TestCase], max_workers: Optional[int] = None) -> List[TestResult]:
        """Run evaluation in parallel for improved performance."""
        workers = max_workers or self.max_workers
        results = []
        total = len(models) * len(test_cases)
        completed = 0
        
        logger.info(f"Starting parallel evaluation with {workers} workers: {len(models)} models x {len(test_cases)} tests x {self.trials} trials = {total * self.trials} total calls")
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_test = {}
            for model in models:
                for test_case in test_cases:
                    future = executor.submit(self.run_single_test, model, test_case)
                    future_to_test[future] = (model, test_case)
            
            for future in as_completed(future_to_test):
                model, test_case = future_to_test[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    status = "✓" if result.passed else "✗"
                    reliability_str = f"({result.reliability:.0f}% reliable)"
                    logger.info(f"[{completed}/{total}] {status} {model} - {test_case.id} {reliability_str}")
                except Exception as e:
                    logger.error(f"Error in parallel execution for {model} - {test_case.id}: {e}")
                    completed += 1
        
        return results
