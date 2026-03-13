"""
AST-based validation for function calling evaluation.
"""

import ast
import json
from typing import List, Dict, Any, Tuple


class ASTValidator:
    """Validates function calls using AST-based substring matching."""
    
    @staticmethod
    def evaluate_call(expected: Dict[str, Any], actual: Dict[str, Any]) -> bool:
        """
        Evaluate if an actual function call matches expected.
        
        Args:
            expected: Expected function call with name and arguments
            actual: Actual function call from model response
        
        Returns:
            True if calls match, False otherwise
        """
        if expected.get("name") != actual.get("name"):
            return False
        
        expected_args = expected.get("arguments", {})
        actual_args = actual.get("arguments", {})
        
        # Check all expected arguments are present with correct values
        for key, value in expected_args.items():
            if key not in actual_args:
                return False
            if actual_args[key] != value:
                return False
        
        return True
    
    @staticmethod
    def evaluate_test(expected_calls: List[Dict[str, Any]], actual_calls: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """
        Evaluate a test case against actual function calls.
        
        Args:
            expected_calls: List of expected function calls
            actual_calls: List of actual function calls from model
        
        Returns:
            Tuple of (passed, message)
        """
        if len(expected_calls) == 0:
            if len(actual_calls) > 0:
                return False, f"Expected no calls but got {len(actual_calls)} call(s)"
            return True, "Correctly abstained from calling any function"
        
        if len(expected_calls) != len(actual_calls):
            return False, f"Expected {len(expected_calls)} call(s), got {len(actual_calls)}"
        
        for i, (expected, actual) in enumerate(zip(expected_calls, actual_calls)):
            if not ASTValidator.evaluate_call(expected, actual):
                exp_str = json.dumps(expected)
                act_str = json.dumps(actual)
                return False, f"Call {i+1} mismatch: expected {exp_str}, got {act_str}"
        
        return True, "All function calls match expected"
