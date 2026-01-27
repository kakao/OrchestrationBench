from typing import Dict, List, Any, Optional, Callable
import json

import aiohttp
from loguru import logger

from src.models.base_model import BaseModel

class OpenSourceModel(BaseModel):
    """OpenSource model implementation for API-based models."""
    
    def __init__(
        self,
        model_name: str = "llama3.1",
        temperature: float = 0.1,
        max_tokens: int = 10240,
        base_url: str = "http://localhost:11434",
        api_key: Optional[str] = None,
        token_callback: Optional[Callable[[str, int, int, float], None]] = None,
        reasoning_effort: str = None,
        **kwargs
    ):
        """Initialize OpenSource model."""
        super().__init__(model_name, temperature, max_tokens, token_callback, **kwargs)
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = None
        
        # Common API formats
        self.api_format = kwargs.get('api_format', 'openai')
        self.reasoning_effort = reasoning_effort 

    async def initialize(self) -> None:
        """Initialize the HTTP session."""
        timeout = aiohttp.ClientTimeout(total= 360)
        self.session = aiohttp.ClientSession(timeout=timeout)
        
        # Test the connection
        try:
            await self.check_availability()
            self.initialized = True
            logger.info(f"OpenSource model {self.model_name} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenSource model: {e}")
            raise
    
    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate a response using the OpenSource model API."""
        if not self.session:
            raise RuntimeError("Model not initialized")
        
        try:
            return await self._generate_openai_compatible(prompt, system_prompt, **kwargs)
                
        except Exception as e:
            logger.error(f"Error generating OpenSource response: {e}")
            raise
    
    async def generate_chat_response(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Generate a chat response using the OpenSource model API."""
        if not self.session:
            raise RuntimeError("Model not initialized")
        
        return await self._chat_openai_compatible(messages, **kwargs)
        
    
    async def _generate_openai_compatible(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate response using OpenAI-compatible API format."""
        url = f"{self.base_url}/chat/completions"
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        return await self._chat_openai_compatible(messages, **kwargs)
    
    async def _chat_openai_compatible(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate chat response using OpenAI-compatible API format."""
        url = f"{self.base_url}/chat/completions"
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        logger.debug(f"Sending request to {url} with headers {headers} and payload {messages}")
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.reasoning_effort is not None:
            kwargs['reasoning_effort'] = self.reasoning_effort
        # if kwargs:
        payload.update({
            k: v for k, v in kwargs.items()
        })
        async with self.session.post(url, json=payload, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"API request failed with status {response.status}: {error_text}")
                raise RuntimeError(f"API request failed with status {response.status}")
            
            data = await response.json()
            logger.debug(f"Response data: {data}")
            content = data["choices"][0]["message"]["content"]
            
            # Update statistics
            usage = data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            
            # If no usage info, estimate
            if not input_tokens and not output_tokens:
                input_tokens = sum(len(msg["content"].split()) for msg in messages) * 1.3
                output_tokens = len(content.split()) * 1.3
                
            self._update_stats(int(input_tokens), int(output_tokens))
            return data["choices"][0]["message"]
    
    async def check_availability(self) -> bool:
        """Check if the OpenSource model API is available."""
        if not self.session:
            return False
        
        try:
            if self.api_format == 'openai':
                url = f"{self.base_url}/v1/models"
                headers = {}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                async with self.session.get(url, headers=headers) as response:
                    return response.status == 200
            else:
                # For custom format, try a simple health check
                url = f"{self.base_url.replace('/v1','')}/health"
                async with self.session.get(url) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.warning(f"OpenSource model availability check failed: {e}")
            return False
    
    async def close(self) -> None:
        """Close the HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
            self.initialized = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
