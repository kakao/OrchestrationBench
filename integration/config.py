"""
Configuration classes for OrchestrationBench integration.
Supports multiple providers: OpenAI, Claude, Gemini, Bedrock, Opensource.
"""
import os
from enum import Enum
from typing import Literal
from pydantic import model_validator, Field
from pydantic_settings import BaseSettings


class Provider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    CLAUDE = "claude"
    GEMINI = "gemini"
    BEDROCK = "bedrock"
    OPENSOURCE = "opensource"


def get_env(env_name: str) -> str | None:
    """
    Get the environment variable value with env_name.
    Supports both ${ENV_NAME} and plain ENV_NAME formats.
    """
    if env_name.startswith("${") and env_name.endswith("}"):
        env_name = env_name[2:-1]

    env_value = os.getenv(env_name)
    if env_value is not None:
        env_value = env_value.strip()
    return env_value


class EnvReplacedSettings(BaseSettings):
    """
    A Pydantic BaseSettings model that recursively replaces values
    like "${ENV_VAR}" with the content of the environment variable.
    """

    @model_validator(mode='before')
    @classmethod
    def _resolve_env_vars(cls, values):
        if isinstance(values, list):
            for value in values:
                if isinstance(value, dict):
                    value = cls._resolve_env_vars(value)
            return values

        if not isinstance(values, dict):
            return values

        for key, value in values.items():
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                env_var_name = value[2:-1]
                resolved = get_env(env_var_name)
                if resolved is not None:
                    values[key] = resolved
        return values


class ModelConfig(EnvReplacedSettings):
    """
    Model configuration supporting multiple providers.

    Provider-specific fields:
    - OpenAI: base_url, model, api_key, temperature
    - Claude: base_url, model, api_key, temperature
    - Gemini: base_url, model, api_key, temperature
    - Bedrock: aws_access_key_id, aws_secret_access_key, aws_region, model
    - Opensource: base_url, model, api_key, temperature
    """
    # Common fields
    provider: Provider = Provider.OPENSOURCE
    model_name: str | None = None
    model_alias: str | None = None
    temperature: float = 0.2

    # API-based provider fields (OpenAI, Claude, Gemini, Opensource)
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None

    # Bedrock-specific fields
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_region: str | None = None

    # Opensource/VLLM specific fields
    model_url: str | None = None  # External vLLM URL (if not starting our own)
    model_path: str | None = None  # Local model path for vLLM

    @model_validator(mode='before')
    @classmethod
    def set_model_alias_if_none(cls, values):
        """If model_alias is not provided, set it to model_name or model."""
        if isinstance(values, list):
            for value in values:
                cls.set_model_alias_if_none(value)
            return values

        if not isinstance(values, dict):
            return values

        model_name = values.get("model_name")
        model = values.get("model")
        model_alias = values.get("model_alias")

        if model_alias is None:
            if model_name:
                values["model_alias"] = model_name
            elif model:
                values["model_alias"] = model

        return values

    def get_effective_model_name(self) -> str:
        """Get the model name to use for API calls."""
        return self.model or self.model_name or self.model_alias

    def to_multiagent_config(self) -> dict:
        """
        Convert to multiagent_config.yaml format based on provider.
        """
        if self.provider == Provider.BEDROCK:
            return {
                "AWS_ACCESS_KEY_ID": self.aws_access_key_id,
                "AWS_SECRET_ACCESS_KEY": self.aws_secret_access_key,
                "AWS_REGION": self.aws_region,
                "model": self.get_effective_model_name(),
            }
        else:
            config = {
                "base_url": self.base_url or self.model_url,
                "model": self.get_effective_model_name(),
                "api_key": self.api_key or "EMPTY",
                "temperature": self.temperature,
                "provider": self.provider.value,
            }
            return config


class VLLMConfig(EnvReplacedSettings):
    """Configuration for vLLM server."""
    port: int = 15142
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.95
    max_model_len: int | None = None
    reasoning_parser: str | None = None
    tool_call_parser: str | None = None
    extra_args: list[str] | None = None


class HfHugLogArgs(BaseSettings):
    """HuggingFace Hub logging arguments."""
    hub_results_org: str = "ovis-kakao"
    hub_repo_name: str = "lm-eval-results"
    push_results_to_hub: bool = True
    push_samples_to_hub: bool = False
    public_repo: bool = False


class JudgeModelSettings(EnvReplacedSettings):
    """Judge model settings."""
    provider: str = "openai"  # openai, claude, gemini, bedrock
    base_url: str = "https://api.openai.com/v1/chat/completions"
    model: str = "gpt-4.1"
    api_format: str = "openai"
    api_key: str | None = None
    # AWS Bedrock credentials
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_region: str | None = None


class JudgeGenerationParams(BaseSettings):
    """Generation parameters for judge model."""
    temperature: float = 0.3
    max_tokens: int = 12288
    top_p: float = 1.0


class JudgeConfig(EnvReplacedSettings):
    """Configuration for judge model."""
    model: JudgeModelSettings = Field(default_factory=JudgeModelSettings)
    generation_params: JudgeGenerationParams = Field(default_factory=JudgeGenerationParams)


class BenchmarkConfig(EnvReplacedSettings):
    """Benchmark execution configuration."""
    temperature: float = 0.2
    num_iter: int = 3  # Number of iterations per scenario
    batch_size: int = 50  # Concurrent agent executions
    max_retries: int = 10  # Max retry attempts for failed executions
    log_level: str = "INFO"  # Logging level for subprocesses
    hf_hug_log_args: HfHugLogArgs | None = None
