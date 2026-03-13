"""
BFCL v4-inspired Test Suite for Function Calling Evaluation.

Implements 30 unique test cases covering single_turn, multi_turn, and agentic scenarios.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


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


class BFCLTestSuite:
    """
    Implements 30 unique test cases inspired by BFCL v4 methodology.
    Covers single_turn (simple, multiple, parallel, parallel_multiple, relevance),
    multi_turn (base, missing_params, missing_functions),
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
