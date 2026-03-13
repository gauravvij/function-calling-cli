"""
OpenRouter API Client for function calling evaluation.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional

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
