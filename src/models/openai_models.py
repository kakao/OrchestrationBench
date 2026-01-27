"""
OpenAI model implementation.

This module provides an interface to OpenAI's models.
"""

from typing import Dict, List, Optional, Callable

import openai
from loguru import logger
from src.models.base_model import BaseModel


class OpenAIModel(BaseModel):
    """OpenAI model implementation."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        openai_api_key: Optional[str] = None,
        token_callback: Optional[Callable[[str, int, int, float], None]] = None,
        **kwargs
    ):
        """Initialize OpenAI model."""
        super().__init__(model_name, temperature, max_tokens, token_callback, **kwargs)
        self.client: Optional[openai.AsyncOpenAI] = None
        self.api_key = openai_api_key
    
    async def initialize(self) -> None:
        """Initialize the OpenAI client."""
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.client = openai.AsyncOpenAI(
            api_key=self.api_key
        )
        
        # Test the connection
        try:
            await self.check_availability()
            self.initialized = True
            logger.info(f"OpenAI model {self.model_name} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI model: {e}")
            raise
    
    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate a response using OpenAI's chat completion."""
        if not self.client:
            raise RuntimeError("Model not initialized")
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        return await self.generate_chat_response(messages, **kwargs)
    
    async def generate_chat_response(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Generate a chat response using OpenAI's API."""
        if not self.client:
            raise RuntimeError("Model not initialized")
        
        # Check budget before making request
        if not self.check_budget():
            budget_info = self.get_budget_info()
            raise RuntimeError(f"Budget limit exceeded. Used: {budget_info['budget_used']}, Limit: {budget_info['budget_limit']}")
        
        # Merge parameters
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.additional_params,
            **kwargs,
        }
        if "gpt-5" in self.model_name:
            params["reasoning_effort"] = "medium"
            params.pop("max_tokens")
            params["max_completion_tokens"] = self.max_tokens
            params.pop("temperature")
        logger.debug(f"OpenAI input messages: {messages}")

        try:
            response = await self.client.chat.completions.create(**params)
                        
            # Update statistics with detailed token information
            if response.usage:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                self._update_stats(input_tokens, output_tokens)
                
                # Log budget warning if approaching limit
                budget_info = self.get_budget_info()
                if budget_info.get("is_budget_warning"):
                    logger.warning(f"Budget warning: {budget_info['percentage_used']}% used ({budget_info['budget_used']}/{budget_info['budget_limit']} tokens)")
            
            return response.choices[0].message
            
        except Exception as e:
            logger.error(f"Error generating OpenAI response: {e}")
            raise
    
    async def check_availability(self) -> bool:
        """Check if OpenAI API is available."""
        if not self.client:
            return False

        try:
            # Make a simple test request
            params = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": "Hello"}],
            }
            # Use max_completion_tokens for gpt-5 models
            if "gpt-5" in self.model_name:
                params["max_completion_tokens"] = 50
            else:
                params["max_tokens"] = 50
            await self.client.chat.completions.create(**params)
            return True
        except Exception as e:
            logger.warning(f"OpenAI availability check failed: {e}")
            return False

