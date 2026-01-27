"""
Base model interface for LLM implementations.

This module defines the abstract interface that all LLM model
implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable

from .pricing import calculate_cost, format_cost


class BaseModel(ABC):
    """
    Abstract base class for LLM model interfaces.
    
    All model implementations (OpenAI, Anthropic, local models)
    must implement this interface to be used by agents.
    """
    
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        token_callback: Optional[Callable[[str, int, int, float], None]] = None,
        **kwargs
    ):
        """Initialize the base model."""
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.additional_params = kwargs
        
        # Token tracking callback
        self.token_callback = token_callback
        
        # Model state
        self.initialized = False
        self.requests_made = 0
        
        # Detailed token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens_used = 0  # Keep for backward compatibility
        self.total_cost = 0.0  # Total cost in USD
        
        # Budget tracking
        self.budget_limit = None  # Token budget limit
        self.budget_used = 0  # Tokens used against budget
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the model connection."""
        pass
    
    @abstractmethod
    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: The input prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        pass
    
    @abstractmethod
    async def generate_chat_response(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Generate a response for a conversation.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        pass
    
    @abstractmethod
    async def check_availability(self) -> bool:
        """Check if the model is available and responsive."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model usage statistics."""
        budget_info = self.get_budget_info()
        stats = {
            "model_name": self.model_name,
            "requests_made": self.requests_made,
            "total_tokens_used": self.total_tokens_used,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "avg_tokens_per_request": (
                self.total_tokens_used / max(self.requests_made, 1)
            ),
            "total_cost": self.total_cost,
            "total_cost_formatted": format_cost(self.total_cost),
            "avg_cost_per_request": (
                self.total_cost / max(self.requests_made, 1)
            ),
            "initialized": self.initialized,
        }
        
        # Add budget information
        stats.update({
            "budget": budget_info
        })
        
        return stats
    
    def _update_stats(self, input_tokens: int, output_tokens: int) -> None:
        """Update usage statistics with detailed token information."""
        self.requests_made += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        total_tokens = input_tokens + output_tokens
        self.total_tokens_used += total_tokens
        
        # Update budget usage
        if hasattr(self, "budget_used"):
            self.budget_used += total_tokens
        
        # Calculate and update total cost
        cost = calculate_cost(
            model_name=self.model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
        self.total_cost += cost
        
        # Call token tracking callback if provided
        if self.token_callback:
            self.token_callback(self.model_name, input_tokens, output_tokens, cost)
    
    def get_usage_info(self) -> Dict[str, Any]:
        """Get detailed cost and usage information."""
        return {
            "model_name": self.model_name,
            "requests_made": self.requests_made,
            "tokens": {
                "total": self.total_tokens_used,
                "input": self.total_input_tokens,
                "output": self.total_output_tokens,
                "avg_per_request": self.total_tokens_used / max(self.requests_made, 1)
            },
            "cost": {
                "total": self.total_cost,
                "total_formatted": format_cost(self.total_cost),
                "avg_per_request": self.total_cost / max(self.requests_made, 1),
                "avg_per_request_formatted": format_cost(
                    self.total_cost / max(self.requests_made, 1)
                )
            },
            "initialized": self.initialized,
        }
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(model={self.model_name}, initialized={self.initialized})>"
    
    def check_budget(self) -> bool:
        """Check if the current usage is within the allowed budget."""
        # If budget_limit is not set, always allow
        if not hasattr(self, "budget_limit") or self.budget_limit is None:
            return True
        # If budget_used is not set, treat as zero
        if not hasattr(self, "budget_used") or self.budget_used is None:
            return True
        return self.budget_used < self.budget_limit
    
    def set_budget_limit(self, token_limit: int) -> None:
        """Set the token budget limit."""
        self.budget_limit = token_limit
        
    def get_budget_info(self) -> Dict[str, Any]:
        """Get detailed budget information."""
        if not hasattr(self, "budget_limit") or self.budget_limit is None:
            return {
                "budget_limit": None,
                "budget_used": getattr(self, "budget_used", 0),
                "budget_remaining": None,
                "percentage_used": 0.0,
                "is_budget_warning": False,
                "is_over_budget": False
            }
            
        budget_used = getattr(self, "budget_used", 0)
        budget_remaining = max(0, self.budget_limit - budget_used)
        percentage_used = (budget_used / self.budget_limit) * 100 if self.budget_limit > 0 else 0.0
        
        return {
            "budget_limit": self.budget_limit,
            "budget_used": budget_used,
            "budget_remaining": budget_remaining,
            "percentage_used": percentage_used,
            "is_budget_warning": percentage_used >= 80.0,  # Warning at 80%
            "is_over_budget": budget_used >= self.budget_limit
        }
    
    def reset_budget_usage(self) -> None:
        """Reset the budget usage counter."""
        self.budget_used = 0


