"""
CLI entry point for OrchestrationBench evaluation.

Usage:
    uv run evaluate config/openai.yaml
    uv run evaluate config/claude.yaml --experiment-id my_exp
"""
import sys
from datetime import datetime
from pathlib import Path

import typer
from loguru import logger
from omegaconf import OmegaConf

from integration.config import (
    ModelConfig,
    BenchmarkConfig,
    JudgeConfig,
    JudgeModelSettings,
    JudgeGenerationParams,
    VLLMConfig,
    HfHugLogArgs,
    Provider,
)
from integration.task import OrchestrationBenchTask
from src.utils.config_loader import load_dotenv

app = typer.Typer(
    name="evaluate",
    help="Run OrchestrationBench evaluation with specified configuration.",
    add_completion=False,
)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file with Hydra-style defaults support."""
    import os

    # Load .env file to make environment variables available for config resolution
    load_dotenv()

    config_dir = config_path.parent

    # Load main config
    main_config = OmegaConf.load(config_path)

    # Process defaults if present
    defaults = OmegaConf.select(main_config, "defaults") or []

    # Load and merge default configs
    merged_config = OmegaConf.create({})
    for default in defaults:
        if isinstance(default, str) and default != "_self_":
            default_path = config_dir / f"{default}.yaml"
            if default_path.exists():
                default_config = OmegaConf.load(default_path)
                merged_config = OmegaConf.merge(merged_config, default_config)

    # Remove defaults key and merge main config (allows override)
    if "defaults" in main_config:
        del main_config["defaults"]

    merged_config = OmegaConf.merge(merged_config, main_config)

    # Register environment variable resolver
    if not OmegaConf.has_resolver("env"):
        OmegaConf.register_new_resolver("env", lambda x: os.getenv(x, ""))

    # Convert to container without resolving (resolve manually later)
    return OmegaConf.to_container(merged_config, resolve=False)


def parse_model_config(config: dict) -> ModelConfig:
    """Parse model configuration from dict."""
    model_section = config.get("model", {})

    # Map provider string to Provider enum
    provider_str = model_section.get("provider", "opensource").lower()
    provider_map = {
        "openai": Provider.OPENAI,
        "claude": Provider.CLAUDE,
        "gemini": Provider.GEMINI,
        "bedrock": Provider.BEDROCK,
        "opensource": Provider.OPENSOURCE,
    }
    provider = provider_map.get(provider_str, Provider.OPENSOURCE)

    return ModelConfig(
        provider=provider,
        model_name=model_section.get("model_name"),
        model_alias=model_section.get("model_alias"),
        model=model_section.get("model"),
        base_url=model_section.get("base_url"),
        api_key=model_section.get("api_key"),
        model_url=model_section.get("model_url"),
        model_path=model_section.get("model_path"),
        temperature=model_section.get("temperature", 0.2),
        # Bedrock specific
        aws_access_key_id=model_section.get("aws_access_key_id"),
        aws_secret_access_key=model_section.get("aws_secret_access_key"),
        aws_region=model_section.get("aws_region"),
    )


def parse_benchmark_config(config: dict) -> BenchmarkConfig:
    """Parse benchmark configuration from dict."""
    benchmark_section = config.get("benchmark", {})

    hf_args = None
    hf_section = benchmark_section.get("hf_hug_log_args")
    if hf_section:
        hf_args = HfHugLogArgs(
            hub_results_org=hf_section.get("hub_results_org", "ovis-kakao"),
            hub_repo_name=hf_section.get("hub_repo_name", "lm-eval-results"),
            push_results_to_hub=hf_section.get("push_results_to_hub", False),
            push_samples_to_hub=hf_section.get("push_samples_to_hub", False),
            public_repo=hf_section.get("public_repo", False),
        )

    return BenchmarkConfig(
        temperature=benchmark_section.get("temperature", 0.2),
        num_iter=benchmark_section.get("num_iter", 3),
        batch_size=benchmark_section.get("batch_size", 50),
        max_retries=benchmark_section.get("max_retries", 10),
        log_level=benchmark_section.get("log_level", "INFO"),
        hf_hug_log_args=hf_args,
    )


def parse_vllm_config(config: dict) -> VLLMConfig | None:
    """Parse vLLM configuration from dict."""
    vllm_section = config.get("vllm")
    if not vllm_section:
        return None

    return VLLMConfig(
        port=vllm_section.get("port", 15142),
        tensor_parallel_size=vllm_section.get("tensor_parallel_size", 1),
        gpu_memory_utilization=vllm_section.get("gpu_memory_utilization", 0.95),
        max_model_len=vllm_section.get("max_model_len"),
        reasoning_parser=vllm_section.get("reasoning_parser"),
        tool_call_parser=vllm_section.get("tool_call_parser"),
        extra_args=vllm_section.get("extra_args"),
    )


def parse_judge_config(config: dict) -> JudgeConfig | None:
    """Parse judge model configuration from dict."""
    judge_section = config.get("judge")
    if not judge_section:
        return None

    model_section = judge_section.get("model", {})
    gen_params_section = judge_section.get("generation_params", {})

    model_settings = JudgeModelSettings(
        provider=model_section.get("provider", "openai"),
        base_url=model_section.get("base_url", "https://api.openai.com/v1/chat/completions"),
        model=model_section.get("model", "gpt-4.1"),
        api_format=model_section.get("api_format", "openai"),
        api_key=model_section.get("api_key"),
        aws_access_key_id=model_section.get("aws_access_key_id"),
        aws_secret_access_key=model_section.get("aws_secret_access_key"),
        aws_region=model_section.get("aws_region"),
    )

    gen_params = JudgeGenerationParams(
        temperature=gen_params_section.get("temperature", 0.3),
        max_tokens=gen_params_section.get("max_tokens", 12288),
        top_p=gen_params_section.get("top_p", 1.0),
    )

    return JudgeConfig(
        model=model_settings,
        generation_params=gen_params,
    )


@app.command()
def main(
    config_path: Path = typer.Argument(
        ...,
        help="Path to the configuration YAML file (e.g., config/openai.yaml)",
        exists=True,
        readable=True,
    ),
    experiment_id: str = typer.Option(
        None,
        "--experiment-id", "-e",
        help="Experiment ID. If not provided, uses timestamp.",
    ),
    results_dir: str = typer.Option(
        None,
        "--results-dir", "-o",
        help="Directory to save results. Defaults to ./results",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging.",
    ),
):
    """
    Run OrchestrationBench evaluation with the specified configuration.

    Example:
        uv run evaluate config/openai.yaml
        uv run evaluate config/claude.yaml --experiment-id my_experiment
        uv run evaluate config/opensource.yaml -o /path/to/results -v
    """
    # Configure logging
    logger.remove()
    log_level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    logger.info(f"Loading configuration from: {config_path}")

    # Load configuration
    config = load_config(config_path)

    # Parse configurations
    model_config = parse_model_config(config)
    benchmark_config = parse_benchmark_config(config)
    vllm_config = parse_vllm_config(config)
    judge_config = parse_judge_config(config)

    # Override log_level if verbose flag is set
    if verbose:
        benchmark_config.log_level = "DEBUG"

    # Set experiment ID
    if not experiment_id:
        experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info(f"Experiment ID: {experiment_id}")
    logger.info(f"Provider: {model_config.provider.value}")
    logger.info(f"Model: {model_config.get_effective_model_name()}")

    # Log judge model info if configured
    if judge_config:
        logger.info(f"Judge Model: {judge_config.model.model}")

    # Create and run task
    task = OrchestrationBenchTask(
        experiment_id=experiment_id,
        model_config=model_config,
        benchmark_config=benchmark_config,
        vllm_config=vllm_config,
        judge_config=judge_config,
        results_dir=results_dir,
    )

    try:
        results = task.run()
        logger.info("Evaluation completed successfully!")
        logger.info(f"Overall Score: {results['results'].get('Overall', 0.0):.4f}")
        return 0
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise typer.Exit(code=1)


def cli():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    cli()
