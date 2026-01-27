"""
OrchestrationBench Integration Package.

This package provides integration with external evaluation systems,
supporting multiple LLM providers:
- OpenAI
- Claude (Anthropic)
- Gemini (Google)
- Bedrock (AWS)
- Opensource (vLLM)

Example usage:
    from integration import (
        OrchestrationBenchTask,
        ModelConfig,
        BenchmarkConfig,
        JudgeConfig,
        VLLMConfig,
        Provider,
    )

    # Model configuration example
    model_config = ModelConfig(
        provider=Provider.OPENAI,
        model_alias="gpt-4.1-mini",
        model="gpt-4.1-mini",
        base_url="https://api.openai.com/v1",
        api_key="${OPENAI_API_KEY}",
        temperature=0.2,
    )

    # Benchmark configuration
    benchmark_config = BenchmarkConfig(
        temperature=0.2,
        num_iter=3,
        batch_size=50,
        max_retries=10,
    )

    # Optional: Judge model configuration (overrides default)
    judge_config = JudgeConfig(...)

    task = OrchestrationBenchTask(
        experiment_id="exp_001",
        model_config=model_config,
        benchmark_config=benchmark_config,
        judge_config=judge_config,  # Optional
    )

    results = task.run()
"""

from integration.config import (
    ModelConfig,
    BenchmarkConfig,
    JudgeConfig,
    VLLMConfig,
    HfHugLogArgs,
    Provider,
)
from integration.task import OrchestrationBenchTask
from integration.vllm import VLLMServer
from integration.utils import (
    save_format_like_lm_eval_style,
    upload_result_to_huggingface,
)
from integration.cli import cli

__all__ = [
    "cli",
    "ModelConfig",
    "BenchmarkConfig",
    "JudgeConfig",
    "VLLMConfig",
    "HfHugLogArgs",
    "Provider",
    "OrchestrationBenchTask",
    "VLLMServer",
    "save_format_like_lm_eval_style",
    "upload_result_to_huggingface",
]
