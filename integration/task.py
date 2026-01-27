"""
OrchestrationBench evaluation task for external integration.
Supports multiple providers: OpenAI, Claude, Gemini, Bedrock, Opensource.
"""
import os
import sys
import json
import subprocess
from pathlib import Path

from loguru import logger

from integration.config import (
    ModelConfig,
    BenchmarkConfig,
    JudgeConfig,
    VLLMConfig,
    Provider,
)
from integration.vllm import VLLMServer
from integration.utils import save_format_like_lm_eval_style, upload_result_to_huggingface


class OrchestrationBenchTask:
    """
    OrchestrationBench evaluation task.
    Evaluates LLM orchestration capabilities in multi-domain scenarios.
    Runs both Korean and English evaluations.

    Supports multiple providers:
    - OpenAI: Uses OpenAI API
    - Claude: Uses Anthropic API
    - Gemini: Uses Google Generative AI API
    - Bedrock: Uses AWS Bedrock
    - Opensource: Uses vLLM server (local or external)
    """

    def __init__(
        self,
        experiment_id: str,
        model_config: ModelConfig,
        benchmark_config: BenchmarkConfig,
        vllm_config: VLLMConfig | None = None,
        judge_config: JudgeConfig | None = None,
        results_dir: str | None = None,
    ):
        self.experiment_id = experiment_id
        self.model_config = model_config
        self.benchmark_config = benchmark_config
        self.vllm_config = vllm_config or VLLMConfig()
        self.judge_config = judge_config

        # OrchestrationBench directory (parent of integration)
        self.orchestration_bench_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )

        # Save results directory
        if results_dir:
            self.save_dir = os.path.join(
                results_dir,
                model_config.model_alias,
                experiment_id,
                "orchestration_bench"
            )
        else:
            self.save_dir = os.path.join(
                self.orchestration_bench_dir,
                "results",
                model_config.model_alias,
                experiment_id
            )

        os.makedirs(self.save_dir, exist_ok=True)

        # Determine if we need to start vLLM (only for opensource provider without external URL)
        self.use_vllm = (
            model_config.provider == Provider.OPENSOURCE and
            model_config.model_url is None and
            model_config.model_path is not None and
            vllm_config is not None
        )

        if self.use_vllm:
            vllm_log_file_path = os.path.join(self.save_dir, "vllm_server.log")
            self.vllm_server = VLLMServer(model_config, self.vllm_config, vllm_log_file_path)
            self.effective_url = self.vllm_server.get_api_base_url()
        else:
            self.vllm_server = None
            self.effective_url = self._get_effective_url()

    def _get_effective_url(self) -> str | None:
        """Get the effective API URL based on provider."""
        if self.model_config.provider == Provider.OPENSOURCE:
            return self.model_config.model_url or self.model_config.base_url
        elif self.model_config.provider == Provider.BEDROCK:
            return None  # Bedrock doesn't use URL
        else:
            return self.model_config.base_url

    def _get_model_env_vars(self) -> dict:
        """
        Get environment variables for model configuration.
        These are used to override config file settings via environment.
        """
        env = os.environ.copy()
        provider = self.model_config.provider
        model_alias = self.model_config.model_alias

        # Common settings
        env["ORCHESTRATION_BENCH_MODEL_ALIAS"] = model_alias
        env["ORCHESTRATION_BENCH_MODEL"] = self.model_config.get_effective_model_name()
        env["ORCHESTRATION_BENCH_PROVIDER"] = provider.value
        env["ORCHESTRATION_BENCH_TEMPERATURE"] = str(self.benchmark_config.temperature)

        if provider == Provider.BEDROCK:
            if self.model_config.aws_access_key_id:
                env["AWS_ACCESS_KEY_ID"] = self.model_config.aws_access_key_id
            if self.model_config.aws_secret_access_key:
                env["AWS_SECRET_ACCESS_KEY"] = self.model_config.aws_secret_access_key
            if self.model_config.aws_region:
                env["AWS_REGION"] = self.model_config.aws_region
                env["ORCHESTRATION_BENCH_AWS_REGION"] = self.model_config.aws_region
        else:
            # For API-based providers (OpenAI, Claude, Gemini, Opensource)
            if provider == Provider.OPENSOURCE:
                api_key = self.model_config.api_key or "EMPTY"
                base_url = self.effective_url
            else:
                api_key = self.model_config.api_key
                base_url = self.model_config.base_url

            if base_url:
                env["ORCHESTRATION_BENCH_BASE_URL"] = base_url
            if api_key:
                env["ORCHESTRATION_BENCH_API_KEY"] = api_key

        logger.info(f"Model config via environment variables:")
        logger.info(f"  Model Alias: {model_alias}")
        logger.info(f"  Model: {env.get('ORCHESTRATION_BENCH_MODEL')}")
        logger.info(f"  Provider: {provider.value}")
        if provider != Provider.BEDROCK:
            logger.info(f"  Base URL: {env.get('ORCHESTRATION_BENCH_BASE_URL', 'N/A')}")

        return env

    def _get_judge_env_vars(self, env: dict) -> dict:
        """
        Add judge model environment variables to the given env dict.
        These are used to override eval_config.yaml settings via environment.
        """
        if not self.judge_config:
            logger.info("No judge config provided, using defaults from eval_config.yaml")
            return env

        env["ORCHESTRATION_BENCH_JUDGE_PROVIDER"] = self.judge_config.model.provider
        env["ORCHESTRATION_BENCH_JUDGE_MODEL"] = self.judge_config.model.model
        env["ORCHESTRATION_BENCH_JUDGE_BASE_URL"] = self.judge_config.model.base_url or ""
        env["ORCHESTRATION_BENCH_JUDGE_API_FORMAT"] = self.judge_config.model.api_format

        if self.judge_config.model.api_key:
            env["ORCHESTRATION_BENCH_JUDGE_API_KEY"] = self.judge_config.model.api_key
            # Also set OPENAI_API_KEY for backward compatibility
            env["OPENAI_API_KEY"] = self.judge_config.model.api_key

        # AWS Bedrock credentials for judge
        if self.judge_config.model.aws_access_key_id:
            env["ORCHESTRATION_BENCH_JUDGE_AWS_ACCESS_KEY_ID"] = self.judge_config.model.aws_access_key_id
        if self.judge_config.model.aws_secret_access_key:
            env["ORCHESTRATION_BENCH_JUDGE_AWS_SECRET_ACCESS_KEY"] = self.judge_config.model.aws_secret_access_key
        if self.judge_config.model.aws_region:
            env["ORCHESTRATION_BENCH_JUDGE_AWS_REGION"] = self.judge_config.model.aws_region

        env["ORCHESTRATION_BENCH_JUDGE_TEMPERATURE"] = str(self.judge_config.generation_params.temperature)
        env["ORCHESTRATION_BENCH_JUDGE_MAX_TOKENS"] = str(self.judge_config.generation_params.max_tokens)
        env["ORCHESTRATION_BENCH_JUDGE_TOP_P"] = str(self.judge_config.generation_params.top_p)

        logger.info(f"Judge config via environment variables:")
        logger.info(f"  Judge Provider: {self.judge_config.model.provider}")
        logger.info(f"  Judge Model: {self.judge_config.model.model}")
        logger.info(f"  Base URL: {self.judge_config.model.base_url}")
        if self.judge_config.model.aws_access_key_id:
            logger.info(f"  AWS Region: {self.judge_config.model.aws_region}")

        return env

    def _run_subprocess_with_streaming(
        self,
        cmd: list[str],
        log_path: str,
        prefix: str,
        env: dict | None = None,
    ) -> int:
        """
        Run a subprocess with real-time output streaming to both log file and stderr.

        Args:
            cmd: Command to run
            log_path: Path to log file
            prefix: Prefix for each output line (e.g., "[Scenario]")
            env: Environment variables (optional)

        Returns:
            Return code of the subprocess
        """
        process_env = env if env else os.environ.copy()

        # Disable Rich live display features when running as subprocess
        # This prevents "Only one live display may be active at once" errors
        process_env["TERM"] = "dumb"
        process_env["NO_COLOR"] = "1"

        with open(log_path, "w") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.orchestration_bench_dir,
                env=process_env,
                bufsize=1,
            )

            # Stream output to both log file and stderr
            for line in process.stdout:
                # Write to log file
                log_file.write(line)
                log_file.flush()
                # Write to stderr with prefix
                sys.stderr.write(f"{prefix} {line}")
                sys.stderr.flush()

            return_code = process.wait()

        return return_code

    def _run_scenario_processing(self, language: str) -> str:
        """
        Run scenario processing (data generation) for the specified language.
        Returns the output directory path.
        """
        lang_code = "EN" if language == "english" else "KO"
        model_alias = self.model_config.model_alias
        output_dir = os.path.join(self.save_dir, f"step_wise_evaluation_{lang_code}")

        agent_cards_path = os.path.join(
            self.orchestration_bench_dir, "data", lang_code, "multiagent_cards"
        )
        data_path = os.path.join(
            self.orchestration_bench_dir, "data", lang_code, "scenario_data", "*.yaml"
        )

        cmd = [
            "uv", "run", "python", "src/stepwise_scenario_processor.py",
            "--model", model_alias,
            "--agent-cards", agent_cards_path,
            "--data-path", data_path,
            "--num-iter", str(self.benchmark_config.num_iter),
            "--batch-size", str(self.benchmark_config.batch_size),
            "--max-retries", str(self.benchmark_config.max_retries),
            "--output-dir", output_dir,
            "--log-level", self.benchmark_config.log_level,
        ]

        logger.info(f"Running scenario processing for {language}...")
        logger.info(f"  Command: {' '.join(cmd)}")

        log_path = os.path.join(self.save_dir, f"scenario_processing_{lang_code}.log")

        # Get environment with model config
        env = self._get_model_env_vars()

        return_code = self._run_subprocess_with_streaming(
            cmd=cmd,
            log_path=log_path,
            prefix=f"[Scenario-{lang_code}]",
            env=env,
        )

        if return_code != 0:
            logger.error(
                f"Scenario processing for {language} failed with return code {return_code}"
            )
            raise RuntimeError(f"Scenario processing failed for {language}")

        logger.info(f"Scenario processing for {language} completed")
        return output_dir

    def _run_evaluation(self, language: str, input_dir: str) -> dict:
        """
        Run evaluation for the specified language.
        Returns the evaluation results.
        """
        lang_code = "EN" if language == "english" else "KO"
        # Use actual model name for directory path (matches stepwise_scenario_processor.py)
        model_name = self.model_config.get_effective_model_name().replace("/", "_").replace("\\", "_")

        model_output_dir = os.path.join(input_dir, model_name)

        agent_cards_path = os.path.join(
            self.orchestration_bench_dir, "data", lang_code, "multiagent_cards"
        )
        eval_config_path = os.path.join(
            self.orchestration_bench_dir, "config", "base_config", "eval_config.yaml"
        )
        output_file = os.path.join(self.save_dir, f"evaluation_result_{lang_code}.json")
        llm_results_dir = os.path.join(self.save_dir, f"llm_evaluation_logs_{lang_code}")

        cmd = [
            "uv", "run", "python", "src/evaluation.py",
            "--input", model_output_dir,
            "--agent-cards-path", agent_cards_path,
            "--eval-config", eval_config_path,
            "--output", output_file,
            "--save-llm-results",
            "--llm-results-dir", llm_results_dir,
            "--log-level", self.benchmark_config.log_level,
        ]

        logger.info(f"Running evaluation for {language}...")
        logger.info(f"  Command: {' '.join(cmd)}")

        log_path = os.path.join(self.save_dir, f"evaluation_{lang_code}.log")

        # Get environment with judge config
        env = os.environ.copy()
        env = self._get_judge_env_vars(env)

        return_code = self._run_subprocess_with_streaming(
            cmd=cmd,
            log_path=log_path,
            prefix=f"[Eval-{lang_code}]",
            env=env,
        )

        if return_code != 0:
            logger.error(
                f"Evaluation for {language} failed with return code {return_code}"
            )
            raise RuntimeError(f"Evaluation failed for {language}")

        logger.info(f"Evaluation for {language} completed")

        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                return json.load(f)
        else:
            logger.warning(f"Evaluation output file not found: {output_file}")
            return {}

    def _extract_key_metrics(self, eval_result: dict) -> dict:
        """Extract key metrics from the evaluation result."""
        overall_stats = eval_result.get("overall_statistics", {})
        key_metrics = overall_stats.get("key_metrics", {})

        return {
            "Plan": key_metrics.get("Plan", 0.0),
            "Call_Rejection": key_metrics.get("Call Rejection Classification Accuracy", 0.0),
            "FC": key_metrics.get("FC", 0.0),
            "Average": key_metrics.get("Average", 0.0),
        }

    def _convert_result_to_lm_eval_style(self, score_dict: dict) -> dict:
        score_dict_formatted = {f"{key},none": value for key, value in score_dict.items()}
        return save_format_like_lm_eval_style(
            "OrchestrationBench", self.model_config.model_alias, score_dict_formatted, 0
        )

    def run(self) -> dict:
        """
        Run the full OrchestrationBench evaluation.

        Returns:
            dict: Evaluation results with metrics for both languages.
        """
        ret = {
            "experiment_id": self.experiment_id,
            "results": {},
        }

        if self.use_vllm:
            self.vllm_server.start()

        try:
            # Model and judge configs are now passed via environment variables
            # in _run_scenario_processing() and _run_evaluation()

            # Run evaluation for both languages
            languages = ["korean", "english"]
            all_metrics = {}

            for language in languages:
                lang_code = "EN" if language == "english" else "KO"
                logger.info(f"Starting OrchestrationBench evaluation for {language}...")

                # Step 1: Run scenario processing
                output_dir = self._run_scenario_processing(language)

                # Step 2: Run evaluation
                eval_result = self._run_evaluation(language, output_dir)

                # Step 3: Extract key metrics
                metrics = self._extract_key_metrics(eval_result)
                all_metrics[lang_code] = metrics

                logger.info(f"--- OrchestrationBench Results for {language} ---")
                logger.info(f"  Plan: {metrics['Plan']:.4f}")
                logger.info(f"  Call Rejection: {metrics['Call_Rejection']:.4f}")
                logger.info(f"  FC: {metrics['FC']:.4f}")
                logger.info(f"  Average: {metrics['Average']:.4f}")
                logger.info("-" * 50)

            # Calculate combined metrics
            combined_metrics = {
                "Plan_KO": all_metrics.get("KO", {}).get("Plan", 0.0),
                "Plan_EN": all_metrics.get("EN", {}).get("Plan", 0.0),
                "Call_Rejection_KO": all_metrics.get("KO", {}).get("Call_Rejection", 0.0),
                "Call_Rejection_EN": all_metrics.get("EN", {}).get("Call_Rejection", 0.0),
                "FC_KO": all_metrics.get("KO", {}).get("FC", 0.0),
                "FC_EN": all_metrics.get("EN", {}).get("FC", 0.0),
                "Average_KO": all_metrics.get("KO", {}).get("Average", 0.0),
                "Average_EN": all_metrics.get("EN", {}).get("Average", 0.0),
            }

            # Calculate overall averages
            combined_metrics["Plan"] = (
                combined_metrics["Plan_KO"] + combined_metrics["Plan_EN"]
            ) / 2
            combined_metrics["Call_Rejection"] = (
                combined_metrics["Call_Rejection_KO"] + combined_metrics["Call_Rejection_EN"]
            ) / 2
            combined_metrics["FC"] = (
                combined_metrics["FC_KO"] + combined_metrics["FC_EN"]
            ) / 2
            combined_metrics["Overall"] = (
                combined_metrics["Average_KO"] + combined_metrics["Average_EN"]
            ) / 2

            ret["results"] = combined_metrics
            ret["result"] = combined_metrics

            logger.info("=== OrchestrationBench Final Results ===")
            logger.info(f"  Plan: {combined_metrics['Plan']:.4f}")
            logger.info(f"  C/R: {combined_metrics['Call_Rejection']:.4f}")
            logger.info(f"  FC: {combined_metrics['FC']:.4f}")
            logger.info(f"  Overall: {combined_metrics['Overall']:.4f}")
            logger.info("=" * 41)

            # Upload to HuggingFace if configured
            hf_args = self.benchmark_config.hf_hug_log_args
            if hf_args and hf_args.push_results_to_hub:
                logger.info("Uploading to HuggingFace...")
                lm_eval_style_result = self._convert_result_to_lm_eval_style(combined_metrics)
                lm_eval_style_result_path = os.path.join(
                    self.save_dir, "lm_eval_style_result.json"
                )
                with open(lm_eval_style_result_path, "w") as f:
                    json.dump(lm_eval_style_result, f, indent=4, ensure_ascii=False)

                result_hf_path = os.path.join(
                    self.model_config.model_alias,
                    f"results_orchestration_bench_{self.experiment_id}.json"
                )
                upload_result_to_huggingface(
                    hf_args,
                    lm_eval_style_result_path,
                    result_hf_path=result_hf_path
                )
                logger.info("Uploaded to HuggingFace!")

            logger.info(
                f"Completed OrchestrationBench evaluation for {self.model_config.model_alias}"
            )
            return ret

        except Exception as e:
            logger.error(f"OrchestrationBench evaluation failed: {e}")
            ret["status"] = "error"
            ret["error"] = str(e)
            raise

        finally:
            if self.use_vllm and self.vllm_server:
                self.vllm_server.stop()
