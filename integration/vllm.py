"""
vLLM Server management for OrchestrationBench integration.
"""
import os
import subprocess
import signal
import sys
import time
import threading
import requests
from loguru import logger

from integration.config import ModelConfig, VLLMConfig


class VLLMServer:
    """Manages a vLLM API server for serving local models."""

    def __init__(
        self,
        model_config: ModelConfig,
        vllm_config: VLLMConfig,
        log_file_path: str | None = None
    ):
        self.model_config = model_config
        self.vllm_config = vllm_config
        self.port = vllm_config.port
        self.log_file_path = log_file_path or "vllm_server.log"

        self.process = None
        self.log_file = None
        self.max_model_len = None

        # Log tailing
        self._stop_log_tail = threading.Event()
        self._log_tail_thread = None

        signal.signal(signal.SIGCHLD, self._sigchld_handler)

    def _sigchld_handler(self, signum, frame):
        """Handle SIGCHLD signal to prevent zombie processes."""
        try:
            while True:
                pid, status = os.waitpid(-1, os.WNOHANG)
                if pid == 0:
                    break
        except OSError:
            pass

    def _get_vllm_max_model_len(self, model_path: str) -> int | dict:
        """
        Get the model's max_model_len using vLLM's internal logic.
        Only analyzes config without loading model weights.
        """
        try:
            from vllm.engine.arg_utils import EngineArgs

            engine_args = EngineArgs(
                model=model_path,
                trust_remote_code=True,
                disable_log_stats=True
            )
            engine_config = engine_args.create_engine_config()
            return engine_config.model_config.max_model_len
        except Exception as e:
            logger.warning(f"Failed to get max_model_len: {e}")
            return {"error": str(e)}

    def start(self):
        """Start vLLM API server."""
        logger.info("Starting vLLM API server...")

        model_name = self.model_config.model_path or self.model_config.model_name
        if not model_name:
            raise ValueError("model_path or model_name is required for vLLM server")

        self.max_model_len = self._get_vllm_max_model_len(model_name)

        cmd = [
            "vllm", "serve", model_name,
            "--port", str(self.port),
            "--served-model-name", self.model_config.model_alias,
            "--gpu-memory-utilization", str(self.vllm_config.gpu_memory_utilization),
        ]

        if self.vllm_config.tensor_parallel_size and self.vllm_config.tensor_parallel_size > 1:
            cmd.extend(["--tensor-parallel-size", str(self.vllm_config.tensor_parallel_size)])

        if self.vllm_config.max_model_len:
            cmd.extend(["--max-model-len", str(self.vllm_config.max_model_len)])

        if self.vllm_config.reasoning_parser:
            cmd.extend(["--reasoning-parser", self.vllm_config.reasoning_parser])

        if self.vllm_config.tool_call_parser:
            cmd.extend(["--enable-auto-tool-choice", "--tool-call-parser", self.vllm_config.tool_call_parser])

        if self.vllm_config.extra_args:
            cmd.extend(self.vllm_config.extra_args)

        env = os.environ.copy()

        log_file = open(self.log_file_path, "a")
        logger.info(f"Starting vLLM API server. cmd: {' '.join(cmd)}")

        self.process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
            universal_newlines=True,
            preexec_fn=os.setsid,
        )

        self.log_file = log_file
        self._wait_for_server()
        logger.info(f"vLLM API server started on port {self.port}")

    def _tail_log_file(self):
        """Tail the log file and print new lines until stopped."""
        # Wait for log file to be created
        while not self._stop_log_tail.is_set():
            if os.path.exists(self.log_file_path):
                break
            time.sleep(0.1)

        if self._stop_log_tail.is_set():
            return

        try:
            with open(self.log_file_path, "r") as f:
                while not self._stop_log_tail.is_set():
                    line = f.readline()
                    if line:
                        # Print without adding extra newline (line already has one)
                        sys.stderr.write(f"[vLLM] {line}")
                        sys.stderr.flush()
                    else:
                        # No new line, wait a bit
                        time.sleep(0.1)
        except Exception as e:
            logger.warning(f"Log tail thread error: {e}")

    def _start_log_tail(self):
        """Start the log tailing thread."""
        self._stop_log_tail.clear()
        self._log_tail_thread = threading.Thread(target=self._tail_log_file, daemon=True)
        self._log_tail_thread.start()

    def _stop_log_tail_thread(self):
        """Stop the log tailing thread."""
        self._stop_log_tail.set()
        if self._log_tail_thread and self._log_tail_thread.is_alive():
            self._log_tail_thread.join(timeout=1.0)
        self._log_tail_thread = None

    def _wait_for_server(self):
        """Wait for vLLM server to be ready."""
        max_wait_time = 3600  # 1 hour
        wait_interval = 2
        elapsed_time = 0

        # Start tailing the log file
        self._start_log_tail()

        try:
            while elapsed_time < max_wait_time:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"vLLM server process terminated unexpectedly. "
                        f"Check {self.log_file_path} for details."
                    )

                try:
                    response = requests.get(f"http://localhost:{self.port}/health", timeout=5)
                    if response.status_code == 200:
                        logger.info("vLLM server is healthy and ready!")
                        return
                except requests.RequestException:
                    pass

                if elapsed_time and elapsed_time % 60 == 0:
                    logger.info(
                        f"Waiting for vLLM server to be ready... "
                        f"({elapsed_time}s/{max_wait_time}s)"
                    )
                time.sleep(wait_interval)
                elapsed_time += wait_interval

            raise TimeoutError(f"vLLM server failed to become ready within {max_wait_time} seconds.")
        finally:
            # Stop tailing the log file
            self._stop_log_tail_thread()

    def stop(self):
        """Stop vLLM API server safely."""
        if self.process:
            logger.info("Stopping vLLM API server...")
            try:
                if self.process.poll() is None:
                    try:
                        logger.info(f"Shutting down vLLM server (PID: {self.process.pid})...")
                        os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)

                        try:
                            self.process.wait(timeout=15)
                            logger.info("vLLM server stopped gracefully")
                        except subprocess.TimeoutExpired:
                            logger.warning("vLLM server did not stop gracefully, force killing...")
                            try:
                                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                                self.process.wait(timeout=5)
                                logger.info("vLLM server force killed")
                            except (ProcessLookupError, subprocess.TimeoutExpired):
                                logger.warning("Process may have already terminated")

                    except ProcessLookupError:
                        logger.info("vLLM process already terminated")
                else:
                    logger.info("vLLM process already terminated")

                try:
                    self.process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass

            except Exception as e:
                logger.error(f"Error stopping vLLM server: {e}")
            finally:
                if self.log_file:
                    try:
                        self.log_file.close()
                    except:
                        pass
                    self.log_file = None
                self.process = None

    def is_running(self) -> bool:
        """Check if vLLM server is running."""
        return self.process is not None and self.process.poll() is None

    def get_api_base_url(self) -> str:
        """Get the API base URL for the vLLM server."""
        return f"http://localhost:{self.port}/v1"

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure cleanup."""
        self.stop()

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.stop()
        except:
            pass
