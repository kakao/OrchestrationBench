"""
Model Factory for creating appropriate LLM model instances.

This module provides centralized model creation logic based on configuration.
"""

from typing import Dict, Any, Optional, Callable
from loguru import logger
from src.models import OpenAIModel, AnthropicModel, OpenSourceModel
from src.models import GeminiModel
from src.models import BedrockModel


def _detect_provider(llm_config: Dict[str, Any]) -> str:
    """
    Detect the provider from config, either explicit or auto-detected.

    Priority:
    1. Explicit provider field
    2. Auto-detect from URL/model name
    3. Default to openai
    """
    provider = llm_config.get("provider", "").lower()

    if provider in ("gemini", "claude", "openai", "bedrock", "opensource"):
        return provider

    # Auto-detect from config
    base_url = llm_config.get("base_url", "")
    model_name = llm_config.get("model", "")

    if "localhost" in base_url or "127.0.0.1" in base_url:
        return "opensource"
    if model_name.startswith("gemini"):
        return "gemini"
    if llm_config.get("aws_access_key_id") or llm_config.get("AWS_ACCESS_KEY_ID"):
        return "bedrock"
   if "anthropic" in base_url or model_name.startswith("claude"):
        return "claude"

    return "openai"


def _create_model_instance(
    llm_config: Dict[str, Any],
    token_callback: Optional[Callable] = None
) -> Any:
    """Create an uninitialized model instance based on configuration."""
    provider = _detect_provider(llm_config)

    model_name = llm_config.get("model", "gpt-4o")
    api_key = llm_config.get("api_key", "")
    temperature = llm_config.get("temperature", 0.2)

    if provider == "gemini":
        model = GeminiModel(
            model_name=model_name,
            temperature=temperature,
            google_api_key=api_key,
            token_callback=token_callback
        )
    elif provider == "claude":
        model = AnthropicModel(
            model_name=model_name,
            temperature=temperature,
            anthropic_api_key=api_key,
            token_callback=token_callback
        )
    elif provider == "bedrock":
        model = BedrockModel(
            model_name=model_name,
            temperature=temperature,
            aws_access_key_id=llm_config.get("aws_access_key_id"),
            aws_secret_access_key=llm_config.get("aws_secret_access_key"),
            aws_region=llm_config.get("aws_region", "us-east-1"),
            token_callback=token_callback
        )
    elif provider == "opensource":
        model = OpenSourceModel(
            model_name=model_name,
            temperature=temperature,
            base_url=llm_config.get("base_url", ""),
            api_key=api_key,
            api_format=llm_config.get("api_format"),
            token_callback=token_callback,
            reasoning_effort=llm_config.get("reasoning_effort")
        )
    else:  # openai (default)
        model = OpenAIModel(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=api_key,
            token_callback=token_callback
        )

    logger.debug(f"Created {type(model).__name__}: {model_name}")
    return model


class ModelFactory:
    """Factory class for creating LLM model instances based on configuration."""

    @staticmethod
    async def create_model(
        llm_config: Dict[str, Any],
        token_callback: Optional[Callable] = None
    ) -> Any:
        """
        Create and initialize an appropriate LLM model based on configuration.

        Args:
            llm_config: Configuration dictionary containing model settings
            token_callback: Optional callback function for token tracking

        Returns:
            Initialized model instance

        Raises:
            RuntimeError: If model initialization fails
        """
        model = _create_model_instance(llm_config, token_callback)
        model_name = llm_config.get("model", "gpt-4o")

        try:
            await model.initialize()
            logger.debug(f"Successfully initialized model {model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to initialize model {model_name}: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")


def create_model_sync(
    llm_config: Dict[str, Any],
    token_callback: Optional[Callable] = None
) -> Any:
    """
    Synchronous wrapper for model creation (without initialization).

    Args:
        llm_config: Configuration dictionary containing model settings
        token_callback: Optional callback function for token tracking

    Returns:
        Uninitialized model instance
    """
    return _create_model_instance(llm_config, token_callback)
