"""
Report generation for evaluation results.
"""

import os
import json
import logging
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime

from .evaluator import TestResult

logger = logging.getLogger(__name__)


@dataclass
class ModelScore:
    """Aggregated scores for a model across all test categories."""
    model: str
    overall_accuracy: float
    overall_reliability: float
    category_scores: Dict[str, float]
    category_reliability: Dict[str, float]
    subcategory_scores: Dict[str, float]
    subcategory_reliability: Dict[str, float]
    total_tests: int
    passed_tests: int
    avg_latency_ms: float
    total_tokens: int


class ReportGenerator:
    """Generates evaluation reports with comparative metrics including reliability."""
    
    def __init__(self, results: List[TestResult]):
        self.results = results
        self.models = sorted(set(r.model for r in results))
        self.categories = sorted(set(r.category for r in results))
        self.subcategories = sorted(set(r.subcategory for r in results))
    
    def calculate_model_scores(self) -> List[ModelScore]:
        """Calculate aggregated scores for each model including reliability metrics."""
        scores = []
        
        for model in self.models:
            model_results = [r for r in self.results if r.model == model]
            
            total = len(model_results)
            passed = sum(1 for r in model_results if r.passed)
            overall_accuracy = passed / total if total > 0 else 0.0
            
            overall_reliability = sum(r.reliability for r in model_results) / len(model_results) if model_results else 0.0
            
            category_scores = {}
            category_reliability = {}
            for cat in self.categories:
                cat_results = [r for r in model_results if r.category == cat]
                if cat_results:
                    cat_passed = sum(1 for r in cat_results if r.passed)
                    category_scores[cat] = cat_passed / len(cat_results)
                    category_reliability[cat] = sum(r.reliability for r in cat_results) / len(cat_results)
            
            subcategory_scores = {}
            subcategory_reliability = {}
            for subcat in self.subcategories:
                subcat_results = [r for r in model_results if r.subcategory == subcat]
                if subcat_results:
                    subcat_passed = sum(1 for r in subcat_results if r.passed)
                    subcategory_scores[subcat] = subcat_passed / len(subcat_results)
                    subcategory_reliability[subcat] = sum(r.reliability for r in subcat_results) / len(subcat_results)
            
            latencies = [r.avg_latency_ms for r in model_results if r.avg_latency_ms > 0]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
            total_tokens = sum(r.total_tokens for r in model_results)
            
            scores.append(ModelScore(
                model=model,
                overall_accuracy=overall_accuracy,
                overall_reliability=overall_reliability,
                category_scores=category_scores,
                category_reliability=category_reliability,
                subcategory_scores=subcategory_scores,
                subcategory_reliability=subcategory_reliability,
                total_tests=total,
                passed_tests=passed,
                avg_latency_ms=avg_latency,
                total_tokens=total_tokens
            ))
        
        return scores
    
    def generate_text_report(self) -> str:
        """Generate a human-readable text report with reliability metrics."""
        scores = self.calculate_model_scores()
        trials_per_test = self.results[0].trials.__len__() if self.results else 3
        
        lines = []
        lines.append("=" * 100)
        lines.append("FUNCTION CALLING EVALUATION REPORT - Best of N Trials with Reliability Metrics")
        lines.append("=" * 100)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Total Tests: {len(set(r.test_id for r in self.results))}")
        lines.append(f"Models Evaluated: {len(self.models)}")
        lines.append(f"Trials per Test: {trials_per_test}")
        lines.append("")
        
        # Leaderboard
        lines.append("-" * 100)
        lines.append("LEADERBOARD (Ranked by Accuracy, with Reliability)")
        lines.append("-" * 100)
        lines.append(f"{'Rank':<6} {'Model':<50} {'Accuracy':<12} {'Reliability':<12} {'Tests':<10}")
        lines.append("-" * 100)
        
        sorted_scores = sorted(scores, key=lambda x: (x.overall_accuracy, x.overall_reliability), reverse=True)
        for i, score in enumerate(sorted_scores, 1):
            lines.append(f"{i:<6} {score.model:<50} {score.overall_accuracy:>10.1%}  {score.overall_reliability:>10.1f}%  {score.passed_tests}/{score.total_tests}")
        
        lines.append("")
        
        # Category breakdown
        lines.append("-" * 100)
        lines.append("CATEGORY BREAKDOWN (Accuracy | Reliability)")
        lines.append("-" * 100)
        
        for category in self.categories:
            lines.append(f"\n{category.upper()}:")
            for score in sorted_scores:
                cat_score = score.category_scores.get(category, 0.0)
                cat_reliability = score.category_reliability.get(category, 0.0)
                lines.append(f"  {score.model:<50} {cat_score:>6.1%} | {cat_reliability:>5.1f}%")
        
        # Subcategory breakdown
        lines.append("\n" + "-" * 100)
        lines.append("SUBCATEGORY BREAKDOWN")
        lines.append("-" * 100)
        
        for subcategory in self.subcategories:
            lines.append(f"\n{subcategory.upper()}:")
            for score in sorted_scores:
                subcat_score = score.subcategory_scores.get(subcategory, 0.0)
                subcat_reliability = score.subcategory_reliability.get(subcategory, 0.0)
                lines.append(f"  {score.model:<50} {subcat_score:>6.1%} | {subcat_reliability:>5.1f}%")
        
        # Detailed results
        lines.append("\n" + "-" * 100)
        lines.append("DETAILED TEST RESULTS (with Trial Breakdown)")
        lines.append("-" * 100)
        
        for result in sorted(self.results, key=lambda x: (x.model, x.test_id)):
            status = "✓ PASS" if result.passed else "✗ FAIL"
            lines.append(f"\n{status} | {result.model} | {result.test_id} | Reliability: {result.reliability:.0f}%")
            lines.append(f"  Category: {result.category}/{result.subcategory}")
            
            for trial in result.trials:
                trial_status = "✓" if trial.passed else "✗"
                lines.append(f"    Trial {trial.trial_number}: {trial_status} ({trial.latency_ms:.0f}ms)")
            
            if result.error:
                lines.append(f"  Error: {result.error}")
            lines.append(f"  Avg Latency: {result.avg_latency_ms:.0f}ms")
        
        lines.append("\n" + "=" * 100)
        lines.append("END OF REPORT")
        lines.append("=" * 100)
        
        return "\n".join(lines)
    
    def generate_json_report(self) -> Dict[str, Any]:
        """Generate a structured JSON report with reliability metrics."""
        scores = self.calculate_model_scores()
        trials_per_test = self.results[0].trials.__len__() if self.results else 3
        
        return {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_tests": len(set(r.test_id for r in self.results)),
                "models_evaluated": len(self.models),
                "trials_per_test": trials_per_test,
                "evaluation_method": "Best of N (pass if at least one trial succeeds)"
            },
            "summary": {
                "leaderboard": [
                    {
                        "rank": i + 1,
                        "model": score.model,
                        "accuracy": round(score.overall_accuracy, 4),
                        "reliability_percent": round(score.overall_reliability, 2),
                        "passed_tests": score.passed_tests,
                        "total_tests": score.total_tests,
                        "avg_latency_ms": round(score.avg_latency_ms, 2),
                        "total_tokens": score.total_tokens
                    }
                    for i, score in enumerate(sorted(scores, key=lambda x: (x.overall_accuracy, x.overall_reliability), reverse=True))
                ]
            },
            "model_scores": [asdict(score) for score in scores],
            "detailed_results": [
                {
                    "test_id": r.test_id,
                    "model": r.model,
                    "category": r.category,
                    "subcategory": r.subcategory,
                    "passed": r.passed,
                    "reliability_percent": round(r.reliability, 2),
                    "trials": [
                        {
                            "trial_number": t.trial_number,
                            "passed": t.passed,
                            "latency_ms": round(t.latency_ms, 2),
                            "tokens_used": t.tokens_used,
                            "error": t.error
                        }
                        for t in r.trials
                    ],
                    "expected_calls": r.expected_calls,
                    "error": r.error,
                    "avg_latency_ms": round(r.avg_latency_ms, 2),
                    "total_tokens": r.total_tokens
                }
                for r in self.results
            ]
        }
    
    def save_reports(self, output_dir: str = "./results"):
        """Save both text and JSON reports to files."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        text_report = self.generate_text_report()
        text_path = os.path.join(output_dir, f"evaluation_report_{timestamp}.txt")
        with open(text_path, 'w') as f:
            f.write(text_report)
        logger.info(f"Text report saved: {text_path}")
        
        json_report = self.generate_json_report()
        json_path = os.path.join(output_dir, f"evaluation_report_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(json_report, f, indent=2)
        logger.info(f"JSON report saved: {json_path}")
        
        return text_path, json_path
