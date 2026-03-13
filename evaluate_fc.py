#!/usr/bin/env python3
"""
OpenRouter Function Calling Evaluation Script
Inspired by Berkeley Function Calling Leaderboard (BFCL) v4 Methodology

This script evaluates multiple LLMs hosted on OpenRouter for their function-calling
capabilities using AST-based substring matching for validation.

FEATURES:
- Best of N trials support for reliable evaluation
- Reliability metrics showing consistency across trials
- Parallel and sequential execution modes
- Comprehensive reporting with JSON and TXT outputs
- 30 unique test cases covering single_turn, multi_turn, and agentic scenarios
"""

import os
import sys
import json
import time
import asyncio
import logging
import argparse
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import ast
import re
from collections import defaultdict

import requests
from dotenv import load_dotenv

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

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION - Target Models
# ============================================================================

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

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TestCase:
    """Represents a single test case for function calling evaluation."""
    id: str
    category: str
    subcategory: str
    description: str
    tools: List[Dict[str, Any]]
    messages: List[Dict[str, str]]
    expected_calls: List[Dict[str, Any]]
    difficulty: str = "medium"

@dataclass
class TrialResult:
    """Represents the result of a single trial."""
    trial_number: int
    passed: bool
    response: Optional[str]
    parsed_calls: List[Dict[str, Any]]
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
    expected_calls: List[Dict[str, Any]]
    error: Optional[str] = None
    avg_latency_ms: float = 0.0
    total_tokens: int = 0

@dataclass
class ModelScore:
    """Aggregated scores for a model across all test categories."""
    model: str
    overall_accuracy: float
    overall_reliability: float  # Average reliability across all tests
    category_scores: Dict[str, float]
    category_reliability: Dict[str, float]
    subcategory_scores: Dict[str, float]
    subcategory_reliability: Dict[str, float]
    total_tests: int
    passed_tests: int
    avg_latency_ms: float
    total_tokens: int

# ============================================================================
# BFCL V4 TEST SUITE - 30 UNIQUE TEST CASES
# ============================================================================

class BFCLTestSuite:
    """
    Implements 30 unique test cases inspired by BFCL v4 methodology.
    Covers single_turn (simple, multiple, parallel, parallel_multiple, relevance),
    multi_turn (base, missing_params, missing_functions, long_context),
    and agentic (web_search, memory, format_sensitivity) categories.
    """
    
    def __init__(self):
        self.test_cases: List[TestCase] = []
        self._generate_all_tests()
    
    def _generate_all_tests(self):
        """Generate all 30 test cases across categories."""
        self._generate_single_turn_tests()  # 16 tests
        self._generate_multi_turn_tests()    # 8 tests
        self._generate_agentic_tests()       # 6 tests
    
    def _generate_single_turn_tests(self):
        """Generate 16 single-turn function calling tests."""
        
        # === SIMPLE FUNCTION CALLING (4 tests) ===
        simple_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
        
        self.test_cases.append(TestCase(
            id="single_simple_001",
            category="single_turn",
            subcategory="simple",
            description="Simple single function call with required parameter",
            tools=simple_tools,
            messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
            expected_calls=[{"name": "get_weather", "arguments": {"location": "Tokyo"}}],
            difficulty="easy"
        ))
        
        self.test_cases.append(TestCase(
            id="single_simple_002",
            category="single_turn",
            subcategory="simple",
            description="Simple function call with optional parameter specified",
            tools=simple_tools,
            messages=[{"role": "user", "content": "What's the weather in London in celsius?"}],
            expected_calls=[{"name": "get_weather", "arguments": {"location": "London", "unit": "celsius"}}],
            difficulty="easy"
        ))
        
        self.test_cases.append(TestCase(
            id="single_simple_003",
            category="single_turn",
            subcategory="simple",
            description="Simple function call with different unit parameter",
            tools=simple_tools,
            messages=[{"role": "user", "content": "What's the weather in New York in fahrenheit?"}],
            expected_calls=[{"name": "get_weather", "arguments": {"location": "New York", "unit": "fahrenheit"}}],
            difficulty="easy"
        ))
        
        self.test_cases.append(TestCase(
            id="single_simple_004",
            category="single_turn",
            subcategory="simple",
            description="Simple function call with city name containing spaces",
            tools=simple_tools,
            messages=[{"role": "user", "content": "What's the weather in San Francisco?"}],
            expected_calls=[{"name": "get_weather", "arguments": {"location": "San Francisco"}}],
            difficulty="easy"
        ))
        
        # === MULTIPLE FUNCTION SELECTION (4 tests) ===
        multi_tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_products",
                    "description": "Search for products in catalog",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "category": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_order_status",
                    "description": "Check status of an order",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string"}
                        },
                        "required": ["order_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "cancel_order",
                    "description": "Cancel an existing order",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string"},
                            "reason": {"type": "string"}
                        },
                        "required": ["order_id"]
                    }
                }
            }
        ]
        
        self.test_cases.append(TestCase(
            id="single_multiple_001",
            category="single_turn",
            subcategory="multiple",
            description="Select correct function from multiple options - order status",
            tools=multi_tools,
            messages=[{"role": "user", "content": "Where is my order ORD-12345?"}],
            expected_calls=[{"name": "get_order_status", "arguments": {"order_id": "ORD-12345"}}],
            difficulty="medium"
        ))
        
        self.test_cases.append(TestCase(
            id="single_multiple_002",
            category="single_turn",
            subcategory="multiple",
            description="Select different function from same toolset - product search",
            tools=multi_tools,
            messages=[{"role": "user", "content": "Find me some wireless headphones"}],
            expected_calls=[{"name": "search_products", "arguments": {"query": "wireless headphones"}}],
            difficulty="medium"
        ))
        
        self.test_cases.append(TestCase(
            id="single_multiple_003",
            category="single_turn",
            subcategory="multiple",
            description="Select cancel function with optional reason parameter",
            tools=multi_tools,
            messages=[{"role": "user", "content": "Cancel order ORD-67890 because I changed my mind"}],
            expected_calls=[{"name": "cancel_order", "arguments": {"order_id": "ORD-67890", "reason": "I changed my mind"}}],
            difficulty="medium"
        ))
        
        self.test_cases.append(TestCase(
            id="single_multiple_004",
            category="single_turn",
            subcategory="multiple",
            description="Select product search with category filter",
            tools=multi_tools,
            messages=[{"role": "user", "content": "Search for laptops in the electronics category"}],
            expected_calls=[{"name": "search_products", "arguments": {"query": "laptops", "category": "electronics"}}],
            difficulty="medium"
        ))
        
        # === PARALLEL FUNCTION CALLING (4 tests) ===
        self.test_cases.append(TestCase(
            id="single_parallel_001",
            category="single_turn",
            subcategory="parallel",
            description="Call same function multiple times in parallel - 3 cities",
            tools=simple_tools,
            messages=[{"role": "user", "content": "What's the weather in Tokyo, London, and New York?"}],
            expected_calls=[
                {"name": "get_weather", "arguments": {"location": "Tokyo"}},
                {"name": "get_weather", "arguments": {"location": "London"}},
                {"name": "get_weather", "arguments": {"location": "New York"}}
            ],
            difficulty="medium"
        ))
        
        self.test_cases.append(TestCase(
            id="single_parallel_002",
            category="single_turn",
            subcategory="parallel",
            description="Call same function multiple times in parallel - 2 cities",
            tools=simple_tools,
            messages=[{"role": "user", "content": "What's the weather in Paris and Berlin?"}],
            expected_calls=[
                {"name": "get_weather", "arguments": {"location": "Paris"}},
                {"name": "get_weather", "arguments": {"location": "Berlin"}}
            ],
            difficulty="medium"
        ))
        
        self.test_cases.append(TestCase(
            id="single_parallel_003",
            category="single_turn",
            subcategory="parallel",
            description="Call same function multiple times in parallel - 4 cities",
            tools=simple_tools,
            messages=[{"role": "user", "content": "What's the weather in Sydney, Melbourne, Brisbane, and Perth?"}],
            expected_calls=[
                {"name": "get_weather", "arguments": {"location": "Sydney"}},
                {"name": "get_weather", "arguments": {"location": "Melbourne"}},
                {"name": "get_weather", "arguments": {"location": "Brisbane"}},
                {"name": "get_weather", "arguments": {"location": "Perth"}}
            ],
            difficulty="hard"
        ))
        
        # === PARALLEL MULTIPLE FUNCTION CALLING (4 tests) ===
        parallel_multi_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "Get current time for a timezone",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "timezone": {"type": "string"}
                        },
                        "required": ["timezone"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "send_email",
                    "description": "Send an email",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to": {"type": "string"},
                            "subject": {"type": "string"},
                            "body": {"type": "string"}
                        },
                        "required": ["to", "subject", "body"]
                    }
                }
            }
        ]
        
        self.test_cases.append(TestCase(
            id="single_parallel_multi_001",
            category="single_turn",
            subcategory="parallel_multiple",
            description="Call different functions in parallel - email and time",
            tools=parallel_multi_tools,
            messages=[{"role": "user", "content": "Send an email to john@example.com about the meeting, and also tell me the current time in Tokyo"}],
            expected_calls=[
                {"name": "send_email", "arguments": {"to": "john@example.com", "subject": "Meeting", "body": "About the meeting"}},
                {"name": "get_current_time", "arguments": {"timezone": "Tokyo"}}
            ],
            difficulty="hard"
        ))
        
        self.test_cases.append(TestCase(
            id="single_parallel_multi_002",
            category="single_turn",
            subcategory="parallel_multiple",
            description="Call different functions in parallel - time for multiple zones",
            tools=parallel_multi_tools,
            messages=[{"role": "user", "content": "What time is it in New York and London? Also email admin@example.com saying the report is ready."}],
            expected_calls=[
                {"name": "get_current_time", "arguments": {"timezone": "New York"}},
                {"name": "get_current_time", "arguments": {"timezone": "London"}},
                {"name": "send_email", "arguments": {"to": "admin@example.com", "subject": "Report Ready", "body": "The report is ready"}}
            ],
            difficulty="hard"
        ))
        
        self.test_cases.append(TestCase(
            id="single_parallel_multi_003",
            category="single_turn",
            subcategory="parallel_multiple",
            description="Call different functions in parallel - single email and time",
            tools=parallel_multi_tools,
            messages=[{"role": "user", "content": "Email hello@example.com with subject 'Hello' and body 'World', and get time in UTC"}],
            expected_calls=[
                {"name": "send_email", "arguments": {"to": "hello@example.com", "subject": "Hello", "body": "World"}},
                {"name": "get_current_time", "arguments": {"timezone": "UTC"}}
            ],
            difficulty="hard"
        ))
        
        # === RELEVANCE DETECTION (4 tests) ===
        self.test_cases.append(TestCase(
            id="single_relevance_001",
            category="single_turn",
            subcategory="relevance",
            description="Should NOT call any function - irrelevant query (joke)",
            tools=simple_tools,
            messages=[{"role": "user", "content": "Tell me a joke about programming"}],
            expected_calls=[],
            difficulty="easy"
        ))
        
        self.test_cases.append(TestCase(
            id="single_relevance_002",
            category="single_turn",
            subcategory="relevance",
            description="Should call function - relevant query (weather)",
            tools=simple_tools,
            messages=[{"role": "user", "content": "What's the temperature in Paris?"}],
            expected_calls=[{"name": "get_weather", "arguments": {"location": "Paris"}}],
            difficulty="easy"
        ))
        
        self.test_cases.append(TestCase(
            id="single_relevance_003",
            category="single_turn",
            subcategory="relevance",
            description="Should NOT call any function - irrelevant query (poem)",
            tools=simple_tools,
            messages=[{"role": "user", "content": "Write a poem about the ocean"}],
            expected_calls=[],
            difficulty="easy"
        ))
        
        self.test_cases.append(TestCase(
            id="single_relevance_004",
            category="single_turn",
            subcategory="relevance",
            description="Should call function - relevant query (weather with unit)",
            tools=simple_tools,
            messages=[{"role": "user", "content": "Is it hot in Dubai right now?"}],
            expected_calls=[{"name": "get_weather", "arguments": {"location": "Dubai"}}],
            difficulty="easy"
        ))
    
    def _generate_multi_turn_tests(self):
        """Generate 8 multi-turn conversation tests."""
        
        # === BASE MULTI-TURN (2 tests) ===
        booking_tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_flights",
                    "description": "Search for available flights",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "origin": {"type": "string"},
                            "destination": {"type": "string"},
                            "date": {"type": "string", "format": "date"}
                        },
                        "required": ["origin", "destination", "date"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "book_flight",
                    "description": "Book a selected flight",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "flight_id": {"type": "string"},
                            "passenger_name": {"type": "string"},
                            "seat_class": {"type": "string", "enum": ["economy", "business", "first"]}
                        },
                        "required": ["flight_id", "passenger_name"]
                    }
                }
            }
        ]
        
        self.test_cases.append(TestCase(
            id="multi_base_001",
            category="multi_turn",
            subcategory="base",
            description="Multi-turn: search then book flight",
            tools=booking_tools,
            messages=[
                {"role": "user", "content": "Find flights from NYC to London on 2024-12-25"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "search_flights", "arguments": '{"origin": "NYC", "destination": "London", "date": "2024-12-25"}'}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": '{"flights": [{"id": "BA112", "time": "10:00"}, {"id": "BA114", "time": "14:00"}]}'},
                {"role": "user", "content": "Book the 10:00 flight for John Doe in business class"}
            ],
            expected_calls=[{"name": "book_flight", "arguments": {"flight_id": "BA112", "passenger_name": "John Doe", "seat_class": "business"}}],
            difficulty="medium"
        ))
        
        self.test_cases.append(TestCase(
            id="multi_base_002",
            category="multi_turn",
            subcategory="base",
            description="Multi-turn: search then book flight - economy class",
            tools=booking_tools,
            messages=[
                {"role": "user", "content": "Find flights from LA to Tokyo on 2024-12-20"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "search_flights", "arguments": '{"origin": "LA", "destination": "Tokyo", "date": "2024-12-20"}'}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": '{"flights": [{"id": "JL001", "time": "11:00"}, {"id": "JL002", "time": "16:00"}]}'},
                {"role": "user", "content": "Book the 16:00 flight for Jane Smith"}
            ],
            expected_calls=[{"name": "book_flight", "arguments": {"flight_id": "JL002", "passenger_name": "Jane Smith"}}],
            difficulty="medium"
        ))
        
        # === MISSING PARAMETERS (2 tests) ===
        self.test_cases.append(TestCase(
            id="multi_mp_001",
            category="multi_turn",
            subcategory="missing_params",
            description="Multi-turn: missing parameter requires clarification - flight booking",
            tools=booking_tools,
            messages=[
                {"role": "user", "content": "Book a flight for me"}
            ],
            expected_calls=[],
            difficulty="medium"
        ))
        
        self.test_cases.append(TestCase(
            id="multi_mp_002",
            category="multi_turn",
            subcategory="missing_params",
            description="Multi-turn: missing parameter requires clarification - search flights",
            tools=booking_tools,
            messages=[
                {"role": "user", "content": "Find me some flights"}
            ],
            expected_calls=[],
            difficulty="medium"
        ))
        
        # === MISSING FUNCTIONS (2 tests) ===
        self.test_cases.append(TestCase(
            id="multi_mf_001",
            category="multi_turn",
            subcategory="missing_functions",
            description="Multi-turn: unavailable function requires clarification - hotel",
            tools=booking_tools,
            messages=[
                {"role": "user", "content": "Cancel my hotel reservation"}
            ],
            expected_calls=[],
            difficulty="medium"
        ))
        
        self.test_cases.append(TestCase(
            id="multi_mf_002",
            category="multi_turn",
            subcategory="missing_functions",
            description="Multi-turn: unavailable function requires clarification - car rental",
            tools=booking_tools,
            messages=[
                {"role": "user", "content": "Rent a car for me in Miami"}
            ],
            expected_calls=[],
            difficulty="medium"
        ))
    
    def _generate_agentic_tests(self):
        """Generate 6 agentic capability tests."""
        
        # === WEB SEARCH (2 tests) ===
        web_search_tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_webpage",
                    "description": "Fetch content from a specific webpage URL",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "Webpage URL"}
                        },
                        "required": ["url"]
                    }
                }
            }
        ]
        
        self.test_cases.append(TestCase(
            id="agentic_web_001",
            category="agentic",
            subcategory="web_search",
            description="Web search: single search query",
            tools=web_search_tools,
            messages=[{"role": "user", "content": "Search for information about Python programming language"}],
            expected_calls=[{"name": "web_search", "arguments": {"query": "Python programming language"}}],
            difficulty="easy"
        ))
        
        self.test_cases.append(TestCase(
            id="agentic_web_002",
            category="agentic",
            subcategory="web_search",
            description="Web search: current events query",
            tools=web_search_tools,
            messages=[{"role": "user", "content": "Search for latest news about artificial intelligence"}],
            expected_calls=[{"name": "web_search", "arguments": {"query": "latest news artificial intelligence"}}],
            difficulty="easy"
        ))
        
        # === MEMORY (2 tests) ===
        memory_tools = [
            {
                "type": "function",
                "function": {
                    "name": "store_memory",
                    "description": "Store information in memory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "key": {"type": "string"},
                            "value": {"type": "string"}
                        },
                        "required": ["key", "value"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "retrieve_memory",
                    "description": "Retrieve information from memory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "key": {"type": "string"}
                        },
                        "required": ["key"]
                    }
                }
            }
        ]
        
        self.test_cases.append(TestCase(
            id="agentic_memory_001",
            category="agentic",
            subcategory="memory",
            description="Memory: store and retrieve information - favorite color",
            tools=memory_tools,
            messages=[
                {"role": "user", "content": "Remember that my favorite color is blue"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "store_memory", "arguments": '{"key": "favorite_color", "value": "blue"}'}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": '{"status": "stored"}'},
                {"role": "user", "content": "What is my favorite color?"}
            ],
            expected_calls=[{"name": "retrieve_memory", "arguments": {"key": "favorite_color"}}],
            difficulty="medium"
        ))
        
        self.test_cases.append(TestCase(
            id="agentic_memory_002",
            category="agentic",
            subcategory="memory",
            description="Memory: store and retrieve information - phone number",
            tools=memory_tools,
            messages=[
                {"role": "user", "content": "Remember my phone number is 555-1234"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "store_memory", "arguments": '{"key": "phone_number", "value": "555-1234"}'}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": '{"status": "stored"}'},
                {"role": "user", "content": "What is my phone number?"}
            ],
            expected_calls=[{"name": "retrieve_memory", "arguments": {"key": "phone_number"}}],
            difficulty="medium"
        ))
        
        # === FORMAT SENSITIVITY (2 tests) ===
        format_tools = [
            {
                "type": "function",
                "function": {
                    "name": "execute_sql",
                    "description": "Execute SQL query on database",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "call_rest_api",
                    "description": "Make REST API call",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                            "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE"]},
                            "headers": {"type": "object"},
                            "body": {"type": "object"}
                        },
                        "required": ["url", "method"]
                    }
                }
            }
        ]
        
        self.test_cases.append(TestCase(
            id="agentic_format_001",
            category="agentic",
            subcategory="format_sensitivity",
            description="Format sensitivity: SQL query generation - users",
            tools=format_tools,
            messages=[{"role": "user", "content": "Get all users from the database who signed up in the last 30 days"}],
            expected_calls=[{"name": "execute_sql", "arguments": {"query": "SELECT * FROM users WHERE signup_date >= DATE_SUB(NOW(), INTERVAL 30 DAY)"}}],
            difficulty="hard"
        ))
        
        self.test_cases.append(TestCase(
            id="agentic_format_002",
            category="agentic",
            subcategory="format_sensitivity",
            description="Format sensitivity: REST API call generation",
            tools=format_tools,
            messages=[{"role": "user", "content": "Make a GET request to https://api.example.com/users"}],
            expected_calls=[{"name": "call_rest_api", "arguments": {"url": "https://api.example.com/users", "method": "GET"}}],
            difficulty="hard"
        ))
    
    def get_tests_by_category(self, category: Optional[str] = None) -> List[TestCase]:
        """Get test cases filtered by category."""
        if category is None:
            return self.test_cases
        return [t for t in self.test_cases if t.category == category]
    
    def get_all_tests(self) -> List[TestCase]:
        """Get all test cases."""
        return self.test_cases


# ============================================================================
# OPENROUTER API CLIENT
# ============================================================================

class OpenRouterClient:
    """Client for OpenRouter API with function calling support."""
    
    BASE_URL = "https://openrouter.ai/api/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key required. Set OPENROUTER_API_KEY env var.")
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost",
            "X-Title": "Function Calling Evaluation"
        })
    
    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """Send chat completion request with optional function calling."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        try:
            response = self.session.post(
                f"{self.BASE_URL}/chat/completions",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def extract_tool_calls(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tool calls from API response."""
        tool_calls = []
        
        if "choices" in response and len(response["choices"]) > 0:
            choice = response["choices"][0]
            
            if "message" in choice and "tool_calls" in choice["message"]:
                for tc in choice["message"]["tool_calls"]:
                    if tc.get("type") == "function":
                        try:
                            args = json.loads(tc["function"]["arguments"])
                        except (json.JSONDecodeError, KeyError):
                            args = tc["function"].get("arguments", {})
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except:
                                    args = {}
                        
                        tool_calls.append({
                            "name": tc["function"]["name"],
                            "arguments": args
                        })
        
        return tool_calls


# ============================================================================
# EVALUATION LOGIC - AST-Based Validation
# ============================================================================

class ASTValidator:
    """Validates function calls using AST-based substring matching."""
    
    @staticmethod
    def normalize_value(value: Any) -> Any:
        """Normalize a value for comparison."""
        if isinstance(value, str):
            return value.strip().lower()
        elif isinstance(value, (list, tuple)):
            return [ASTValidator.normalize_value(v) for v in value]
        elif isinstance(value, dict):
            return {k.lower() if isinstance(k, str) else k: ASTValidator.normalize_value(v) 
                    for k, v in value.items()}
        return value
    
    @staticmethod
    def validate_arguments(expected: Dict[str, Any], actual: Dict[str, Any]) -> bool:
        """Validate that actual arguments match expected."""
        expected_norm = ASTValidator.normalize_value(expected)
        actual_norm = ASTValidator.normalize_value(actual)
        
        for key, exp_val in expected_norm.items():
            if key not in actual_norm:
                return False
            act_val = actual_norm[key]
            
            if isinstance(exp_val, list) and isinstance(act_val, list):
                if sorted(exp_val) != sorted(act_val):
                    return False
            elif exp_val != act_val:
                return False
        
        return True
    
    @staticmethod
    def evaluate_call(expected: Dict[str, Any], actual: Dict[str, Any]) -> bool:
        """Evaluate if a single function call matches expected."""
        if expected.get("name") != actual.get("name"):
            return False
        
        expected_args = expected.get("arguments", {})
        actual_args = actual.get("arguments", {})
        
        return ASTValidator.validate_arguments(expected_args, actual_args)
    
    @staticmethod
    def evaluate_test(expected_calls: List[Dict], actual_calls: List[Dict]) -> Tuple[bool, str]:
        """Evaluate a test case against expected calls."""
        if not expected_calls:
            if actual_calls:
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


# ============================================================================
# EVALUATION EXECUTOR
# ============================================================================

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


# ============================================================================
# REPORTING MODULE
# ============================================================================

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
        
        lines = []
        lines.append("=" * 100)
        lines.append("FUNCTION CALLING EVALUATION REPORT - Best of N Trials with Reliability Metrics")
        lines.append("=" * 100)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Total Tests: {len(set(r.test_id for r in self.results))}")
        lines.append(f"Models Evaluated: {len(self.models)}")
        lines.append(f"Trials per Test: {self.results[0].trials.__len__() if self.results else 3}")
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


# ============================================================================
# MAIN EVALUATION SCRIPT
# ============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate LLM function calling capabilities on OpenRouter with Best of N trials"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="List of models to evaluate (default: predefined list)"
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
    
    return parser.parse_args()


def main():
    """Main entry point for the evaluation script."""
    args = parse_arguments()
    
    try:
        client = OpenRouterClient(api_key=args.api_key)
        logger.info("OpenRouter client initialized successfully")
    except ValueError as e:
        logger.error(f"Failed to initialize OpenRouter client: {e}")
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


if __name__ == "__main__":
    main()
