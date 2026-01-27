"""
Utility functions for OrchestrationBench integration.
"""
import copy
import time

from loguru import logger
from huggingface_hub import HfApi

from integration.config import HfHugLogArgs


def save_format_like_lm_eval_style(
    benchmark_name: str,
    model_name: str,
    score_dict: dict,
    num_fewshot: int
) -> dict:
    """
    Format results in lm-eval style for consistency with other benchmarks.
    """
    copied_score_dict = copy.deepcopy(score_dict)

    return {
        "results": {
            benchmark_name: copied_score_dict
        },
        "configs": {
            benchmark_name: {
                "num_fewshot": num_fewshot
            }
        },
        "model_name": model_name,
        "date": time.time(),
    }


def retry_api_call(api_func, *args, max_retries: int = 10, retry_delay: int = 60, **kwargs):
    """
    Retry wrapper for HuggingFace API calls with fixed delay.

    Args:
        api_func: The API function to call
        *args: Positional arguments for the API function
        max_retries: Maximum number of retry attempts (default: 10)
        retry_delay: Delay between retries in seconds (default: 60)
        **kwargs: Keyword arguments for the API function

    Returns:
        The result of the API function call

    Raises:
        The last exception encountered if all retries fail
    """
    for attempt in range(max_retries):
        try:
            return api_func(*args, **kwargs)
        except Exception as error:
            logger.warning(f"API call attempt {attempt + 1}/{max_retries} failed: {error}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed API call after {max_retries} attempts")
                raise error


def upload_result_to_huggingface(
    hf_hug_log_args: HfHugLogArgs,
    result_local_path: str,
    result_hf_path: str | None = None
):
    """
    Upload evaluation results to HuggingFace Hub.

    Args:
        hf_hug_log_args: HuggingFace Hub configuration
        result_local_path: Local path to the result file
        result_hf_path: Path in the HuggingFace repo (defaults to result_local_path)
    """
    if result_hf_path is None:
        result_hf_path = result_local_path

    repo_id = f"{hf_hug_log_args.hub_results_org}/{hf_hug_log_args.hub_repo_name}"

    api = HfApi()
    retry_api_call(
        api.upload_file,
        path_or_fileobj=result_local_path,
        path_in_repo=result_hf_path,
        repo_id=repo_id,
        repo_type="dataset"
    )
    logger.info(f"Uploaded result to HuggingFace: {repo_id}/{result_hf_path}")
