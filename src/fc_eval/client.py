"""
OpenRouter and Ollama API Clients for function calling evaluation.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Union

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()


class OpenRouterClient:
    """Client for OpenRouter API with function calling support."""
    
    BASE_URL = "https://openrouter.ai/api/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenRouter client.
        
        Args:
            api_key: OpenRouter API key. If not provided, reads from OPENROUTER_API_KEY env var.
        
        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. "
                "Set OPENROUTER_API_KEY environment variable or pass --api-key."
            )
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://heyneo.so",
            "X-Title": "FC-Eval Function Calling Benchmark"
        }
        
        logger.info("OpenRouter client initialized")
    
    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        Send a chat completion request with optional function calling.
        
        Args:
            model: Model identifier (e.g., "openai/gpt-4")
            messages: List of message dictionaries
            tools: Optional list of tool/function definitions
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        
        Returns:
            API response dictionary
        
        Raises:
            requests.RequestException: If API request fails
        """
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
            response = requests.post(
                f"{self.BASE_URL}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json()
        
        except requests.RequestException as e:
            logger.error(f"API request failed for model {model}: {e}")
            raise
    
    def extract_tool_calls(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract tool calls from API response.
        
        Args:
            response: API response dictionary
        
        Returns:
            List of parsed tool calls with name and arguments
        """
        tool_calls = []
        
        if "choices" not in response or not response["choices"]:
            return tool_calls
        
        message = response["choices"][0].get("message", {})
        
        # Check for tool_calls in the message
        if "tool_calls" in message:
            for tc in message["tool_calls"]:
                if tc.get("type") == "function":
                    func = tc.get("function", {})
                    try:
                        arguments = json.loads(func.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        arguments = {}
                    tool_calls.append({
                        "name": func.get("name"),
                        "arguments": arguments
                    })
        
        # Check for direct function_call (legacy format)
        elif "function_call" in message:
            func = message["function_call"]
            try:
                arguments = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append({
                "name": func.get("name"),
                "arguments": arguments
            })
        
        return tool_calls


class OllamaClient:
    """Client for Ollama local API with function calling support."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Ollama API base URL (default: http://localhost:11434)
        """
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Content-Type": "application/json"
        }
        
        # Test connection
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            logger.info(f"Ollama client initialized successfully (URL: {self.base_url})")
        except requests.RequestException as e:
            logger.warning(f"Could not connect to Ollama at {self.base_url}: {e}")
            logger.warning("Make sure Ollama is running: ollama serve")
    
    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        Send a chat completion request with optional function calling.
        
        Args:
            model: Model name (e.g., "llama3.2")
            messages: List of message dictionaries
            tools: Optional list of tool/function definitions
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        
        Returns:
            API response dictionary in OpenAI-compatible format
        
        Raises:
            requests.RequestException: If API request fails
        """
        # Convert OpenAI format to Ollama format
        ollama_messages = []
        for msg in messages:
            ollama_msg = {
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            }
            ollama_messages.append(ollama_msg)
        
        payload = {
            "model": model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        # Add tools if provided (Ollama supports tools in recent versions)
        if tools:
            # Convert OpenAI tool format to Ollama format
            ollama_tools = []
            for tool in tools:
                if tool.get("type") == "function":
                    func = tool.get("function", {})
                    ollama_tool = {
                        "type": "function",
                        "function": {
                            "name": func.get("name"),
                            "description": func.get("description", ""),
                            "parameters": func.get("parameters", {})
                        }
                    }
                    ollama_tools.append(ollama_tool)
            payload["tools"] = ollama_tools
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                headers=self.headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            ollama_response = response.json()
            
            # Convert Ollama response to OpenAI-compatible format
            return self._convert_to_openai_format(ollama_response, model)
        
        except requests.RequestException as e:
            logger.error(f"Ollama API request failed for model {model}: {e}")
            raise
    
    def _convert_to_openai_format(self, ollama_response: Dict[str, Any], model: str) -> Dict[str, Any]:
        """
        Convert Ollama response to OpenAI-compatible format.
        
        Args:
            ollama_response: Raw Ollama API response
            model: Model name
        
        Returns:
            OpenAI-compatible response dictionary
        """
        message = ollama_response.get("message", {})
        content = message.get("content", "")
        
        # Build OpenAI-compatible response
        openai_response = {
            "id": ollama_response.get("id", "ollama-chat"),
            "object": "chat.completion",
            "created": ollama_response.get("created_at", 0),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "finish_reason": ollama_response.get("done_reason", "stop")
                }
            ],
            "usage": {
                "prompt_tokens": ollama_response.get("prompt_eval_count", 0),
                "completion_tokens": ollama_response.get("eval_count", 0),
                "total_tokens": (ollama_response.get("prompt_eval_count", 0) + 
                                ollama_response.get("eval_count", 0))
            }
        }
        
        # Handle tool calls if present in Ollama response
        tool_calls = message.get("tool_calls", [])
        if tool_calls:
            openai_tool_calls = []
            for i, tc in enumerate(tool_calls):
                if tc.get("type") == "function":
                    func = tc.get("function", {})
                    openai_tool_calls.append({
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": func.get("name"),
                            "arguments": json.dumps(func.get("arguments", {}))
                        }
                    })
            openai_response["choices"][0]["message"]["tool_calls"] = openai_tool_calls
        
        return openai_response
    
    def extract_tool_calls(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract tool calls from API response.
        
        Args:
            response: API response dictionary (OpenAI-compatible format)
        
        Returns:
            List of parsed tool calls with name and arguments
        """
        tool_calls = []
        
        if "choices" not in response or not response["choices"]:
            return tool_calls
        
        message = response["choices"][0].get("message", {})
        
        # Check for tool_calls in the message
        if "tool_calls" in message:
            for tc in message["tool_calls"]:
                if tc.get("type") == "function":
                    func = tc.get("function", {})
                    try:
                        arguments = json.loads(func.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        arguments = {}
                    tool_calls.append({
                        "name": func.get("name"),
                        "arguments": arguments
                    })
        
        # Check for direct function_call (legacy format)
        elif "function_call" in message:
            func = message["function_call"]
            try:
                arguments = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append({
                "name": func.get("name"),
                "arguments": arguments
            })
        
        return tool_calls
