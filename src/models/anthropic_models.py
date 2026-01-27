"""
Anthropic model implementation.

This module provides an interface to Anthropic's models.
"""

import json
from typing import Dict, List, Any, Optional, Callable

import anthropic
from loguru import logger

from src.models.base_model import BaseModel


class AnthropicModel(BaseModel):
    """Anthropic model implementation."""
    
    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        anthropic_api_key: Optional[str] = None,
        token_callback: Optional[Callable[[str, int, int, float], None]] = None,
        **kwargs
    ):
        """Initialize Anthropic model."""
        super().__init__(model_name, temperature, max_tokens, token_callback, **kwargs)
        self.client: Optional[anthropic.AsyncAnthropic] = None
        self.api_key = anthropic_api_key
    
    async def initialize(self) -> None:
        """Initialize the Anthropic client."""
        if not self.api_key:
            raise ValueError("Anthropic API key not provided")

        self.client = anthropic.AsyncAnthropic(
            api_key=self.api_key
        )

        # Test the connection
        is_available = await self.check_availability()
        if not is_available:
            self.client = None
            raise RuntimeError(f"Failed to connect to Anthropic API. Please check your API key.")

        self.initialized = True
        logger.info(f"Anthropic model {self.model_name} initialized successfully")
    
    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response using Anthropic's message API.

        Returns:
            Dict containing content, tool_calls, and stop_reason
        """
        if not self.client:
            raise RuntimeError("Model not initialized")

        # Prepare messages
        messages = [{"role": "user", "content": prompt}]

        return await self.generate_chat_response(messages, system_prompt=system_prompt, **kwargs)
    
    async def generate_chat_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a chat response using Anthropic's API.

        Returns:
            Dict containing:
                - content: text response (str)
                - tool_calls: list of tool calls in OpenAI format (if any)
                - stop_reason: reason for stopping (end_turn, tool_use, max_tokens, etc.)
        """
        if not self.client:
            raise RuntimeError("Model not initialized")

        # Check budget before making request
        if not self.check_budget():
            raise RuntimeError(f"Budget limit exceeded. Used: {self.budget_used}, Limit: {self.budget_limit}")

        # Extract system messages from messages list (Anthropic doesn't support system role in messages)
        filtered_messages = []
        extracted_system_prompts = []
        for msg in messages:
            if msg.get("role") == "system":
                extracted_system_prompts.append(msg.get("content", ""))
            else:
                filtered_messages.append(msg)

        # Combine extracted system prompts with provided system_prompt
        combined_system = ""
        if extracted_system_prompts:
            combined_system = "\n\n".join(extracted_system_prompts)
        if system_prompt:
            combined_system = f"{combined_system}\n\n{system_prompt}" if combined_system else system_prompt

        # Convert tools to Anthropic format (returns None if empty)
        tools = self.convert_openai_tools_to_anthropic(kwargs.get("tools", []))

        # Prepare parameters
        params = {
            "model": self.model_name,
            "messages": filtered_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.additional_params,
        }

        # Only add tools if present (Anthropic doesn't accept empty tools list)
        if tools:
            params["tools"] = tools

        # Add system prompt if combined
        if combined_system:
            params["system"] = combined_system

        try:
            response = await self.client.messages.create(**params)

            # Extract response content
            text_content = ""
            tool_use_blocks = []

            for content_block in response.content:
                if content_block.type == "text":
                    text_content += content_block.text
                elif content_block.type == "tool_use":
                    tool_use_blocks.append(content_block)

            # Update statistics with detailed token information
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            self._update_stats(input_tokens, output_tokens)

            # Log budget warning if approaching limit
            budget_info = self.get_budget_info()
            if budget_info.get("is_budget_warning"):
                logger.warning(f"Budget warning: {budget_info['percentage_used']}% used ({budget_info['budget_used']}/{budget_info['budget_limit']} tokens)")

            # Log stop reason for debugging
            if response.stop_reason == "max_tokens":
                logger.warning(f"Response truncated due to max_tokens limit ({self.max_tokens})")

            # Build result
            result = {
                "role": "assistant",
                "content": text_content,
                "stop_reason": response.stop_reason,
                "tool_calls": None,
            }

            # Convert tool_use to OpenAI format if present
            if tool_use_blocks:
                result["tool_calls"] = self.convert_tool_use_to_openai_format(tool_use_blocks)

            return result

        except anthropic.BadRequestError as e:
            logger.error(f"Anthropic BadRequestError: {e}")
            raise
        except anthropic.AuthenticationError as e:
            logger.error(f"Anthropic AuthenticationError: {e}")
            raise
        except anthropic.RateLimitError as e:
            logger.error(f"Anthropic RateLimitError: {e}")
            raise
        except Exception as e:
            logger.error(f"Error generating Anthropic response: {e}")
            raise
    
    async def check_availability(self) -> bool:
        """Check if Anthropic API is available."""
        if not self.client:
            return False
        
        try:
            # Make a simple test request
            response = await self.client.messages.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1,
            )
            return True
        except Exception as e:
            logger.warning(f"Anthropic availability check failed: {e}")
            return False
    
    @staticmethod
    def convert_to_anthropic_format(message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a message to Anthropic's format.
        
        Args:
            message: Message in OpenAI format
            
        Returns:
            Converted message in Anthropic format
        """
        return {
            "role": message.get("role", "user"),
            "content": message.get("content", "")
        }

    @staticmethod
    def convert_openai_tools_to_anthropic(openai_tools):
        """
        Convert OpenAI tools format to Anthropic tools format.

        OpenAI format:
        {
            "type": "function",
            "function": {
                "name": "...",
                "description": "...",
                "parameters": {...}
            }
        }

        Anthropic format:
        {
            "name": "...",
            "description": "...",
            "input_schema": {...}
        }
        """
        if not openai_tools:
            return None

        anthropic_tools = []

        for tool in openai_tools:
            if tool.get("type") == "function" and "function" in tool:
                func = tool["function"]

                anthropic_tool = {
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {})
                }

                anthropic_tools.append(anthropic_tool)

        return anthropic_tools if anthropic_tools else None

    @staticmethod
    def convert_anthropic_tools_to_openai(anthropic_tools):
        """
        Convert Anthropic tools format to OpenAI tools format.

        Args:
            anthropic_tools: Anthropic format tools list

        Returns:
            OpenAI format tools list
        """
        if not anthropic_tools:
            return []

        openai_tools = []

        for tool in anthropic_tools:
            # Anthropic format: {"name": ..., "description": ..., "input_schema": ...}
            if "name" in tool:
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {})
                    }
                }
                openai_tools.append(openai_tool)

        return openai_tools

    @staticmethod
    def convert_tool_use_to_openai_format(tool_use_blocks):
        """
        Convert Anthropic tool_use response to OpenAI tool_calls format.

        Args:
            tool_use_blocks: List of tool_use content blocks from Anthropic response

        Returns:
            List of tool calls in OpenAI format
        """
        tool_calls = []
        for block in tool_use_blocks:
            tool_calls.append({
                "id": block.id,
                "type": "function",
                "function": {
                    "name": block.name,
                    "arguments": block.input if isinstance(block.input, str) else json.dumps(block.input)
                }
            })
        return tool_calls
