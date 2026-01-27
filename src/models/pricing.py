"""
Model pricing information for cost tracking.
"""

from typing import Dict, Tuple

# Price per token (in USD)
# Format: "model_name": (input_price_per_token, output_price_per_token)

PRICE_PER_TOKEN_MAP: Dict[str, Tuple[float, float]] = {
    # OpenAI Models
    "gpt-4o-2024-08-06": (2.5 / 1000000, 10.0 / 1000000), # 1M tokens: Input $2.50, Output $10.00
    "gpt-4o-mini": (0.15 / 1000000, 0.6 / 1000000), # 1M tokens: Input $0.15, Output $0.60
    "gpt-4-turbo": (10.0 / 1000000, 30.0 / 1000000), # 1M tokens: Input $10.00, Output $30.00 (This refers to gpt-4-turbo-2024-04-09 or similar)
    "gpt-4": (30.0 / 1000000, 60.0 / 1000000), # 1M tokens: Input $30.00, Output $60.00
    "gpt-3.5-turbo": (0.5 / 1000000, 1.5 / 1000000), # 1M tokens: Input $0.50, Output $1.50
    "gpt-4.1-2025-04-14": (2.0 / 1000000, 8.0 / 1000000), # 1M tokens: Input $2.00, Output $8.00 (Based on Azure OpenAI pricing)
    "gpt-4.1-mini-2025-04-14": (0.40 / 1000000, 1.60 / 1000000), # 1M tokens: Input $0.40, Output $1.60 (Based on Azure OpenAI pricing)
    "o4-mini": (1.10 / 1000000, 4.40 / 1000000), # 1M tokens: Input $1.10, Output $4.40 (Based on Azure OpenAI pricing for o4-mini 2025-04-16)
    
    # Anthropic Models
    "claude-4-sonnet-20241022": (3.0 / 1000000, 15.0 / 1000000), # 1M tokens: Input $3.00, Output $15.00
    "claude-3-5-sonnet-20241022": (3.0 / 1000000, 15.0 / 1000000), # 1M tokens: Input $3.00, Output $15.00
    "claude-3-5-haiku-20241022": (0.80 / 1000000, 4.0 / 1000000), # 1M tokens: Input $0.80, Output $4.00
    "claude-3-opus-20240229": (15.0 / 1000000, 75.0 / 1000000), # 1M tokens: Input $15.00, Output $75.00
    "claude-3-sonnet-20240229": (3.0 / 1000000, 15.0 / 1000000), # 1M tokens: Input $3.00, Output $15.00
    "claude-3-haiku-20240307": (0.25 / 1000000, 1.25 / 1000000), # 1M tokens: Input $0.25, Output $1.25
    
    # Google Models
    "gemini-1.5-pro": (1.25 / 1000000, 5.0 / 1000000), # 1M tokens: Input $1.25, Output $5.00 (for prompts <= 200k tokens)
    "gemini-1.5-pro-large-context": (2.50 / 1000000, 15.0 / 1000000), # 1M tokens: Input $2.50, Output $15.00 (for prompts > 200k tokens, based on some sources, verify if this is still the case)
    "gemini-1.5-flash": (0.075 / 1000000, 0.3 / 1000000), # 1M tokens: Input $0.075, Output $0.30 (for prompts <= 128k tokens)
    "gemini-2.5-flash-lite": (0.10 / 1000000, 0.40 / 1000000), # 1M tokens: Input $0.10, Output $0.40 (Based on latest Google AI pricing)
    "gemini-1.0-pro": (0.5 / 1000000, 1.5 / 1000000), # 1M tokens: Input $0.50, Output $1.50
}
# Fallback prices for unknown models
INPUT_PRICE_PER_TOKEN_FALLBACK = 15.0 / 1000000
OUTPUT_PRICE_PER_TOKEN_FALLBACK = 45.0 / 1000000

# Custom/Local models (free)
LOCAL_MODELS = {
    "gemma3", "qwen3"
}


def get_token_price(model_name: str) -> Tuple[float, float]:
    """
    Get token prices for a model.
    
    Args:
        model_name: The model name
        
    Returns:
        Tuple of (input_price_per_token, output_price_per_token)
    """
    # Check if it's a local/custom model
    for local_model in LOCAL_MODELS:
        if local_model in model_name.lower():
            return (0.0, 0.0)
    
    # Check exact match first
    if model_name in PRICE_PER_TOKEN_MAP:
        return PRICE_PER_TOKEN_MAP[model_name]
    
    # Check for partial matches
    for price_model, prices in PRICE_PER_TOKEN_MAP.items():
        if price_model in model_name or model_name in price_model:
            return prices
    
    # Fallback to default pricing
    return (INPUT_PRICE_PER_TOKEN_FALLBACK, OUTPUT_PRICE_PER_TOKEN_FALLBACK)


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate the cost for a model usage.
    
    Args:
        model_name: The model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        
    Returns:
        Total cost in USD
    """
    input_price, output_price = get_token_price(model_name)
    return (input_tokens * input_price) + (output_tokens * output_price)


def format_cost(cost: float) -> str:
    """Format cost for display."""
    if cost == 0:
        return "Free"
    elif cost < 0.01:
        return f"${cost:.6f}"
    else:
        return f"${cost:.4f}"
