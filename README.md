# OrchestrationBench

A comprehensive bilingual (Korean/English) benchmark framework for evaluating Large Language Model (LLM) orchestration capabilities in multi-domain scenarios. This benchmark evaluates how well LLMs can plan complex workflows and execute tools under realistic constraints across 17 representative domains with nearly 100 realistic virtual tools.

## 🎯 Overview

The OrchestrationBench tests the ability of language models to:
- **Workflow-based Planning**: Structure multi-step processes and coordinate across multiple domains using DAG-based evaluation
- **Constraint-aware Tool Execution**: Generate correct function calls with appropriate arguments while handling business constraints and rejection cases

## 📁 Project Structure

```
OrchestrationBench/
├── config/
│   ├── base_config/
│   │   ├── eval_config.yaml       # Default judge model and evaluation settings
│   │   └── multiagent_config.yaml # LLM model configurations and prompts
│   ├── bedrock.yaml               # AWS Bedrock provider config
│   ├── claude.yaml                # Anthropic Claude provider config
│   ├── gemini.yaml                # Google Gemini provider config
│   ├── openai.yaml                # OpenAI provider config
│   ├── opensource_api.yaml        # External vLLM server config
│   └── opensource_vllm.yaml       # Local vLLM server config
├── data/
│   ├── EN/                        # English scenarios and agent cards
│   │   ├── multiagent_cards/      # 17 domain agent definitions (17 agents)
│   │   └── scenario_data/         # Test scenarios (222 YAML files)
│   ├── KO/                        # Korean scenarios and agent cards
│   │   ├── multiagent_cards/      # 17 domain agent definitions (17 agents)
│   │   └── scenario_data/         # Test scenarios (222 YAML files, Korean version)
│   └── results/                   # Generated results and evaluation outputs
├── src/
│   ├── stepwise_scenario_processor.py  # Main scenario execution engine
│   ├── evaluation.py              # Comprehensive evaluation pipeline
│   ├── orchestration_engine.py    # Simplified orchestration engine
│   ├── step_history_generator.py  # History generation logic
│   ├── agents/                    # Agent implementations
│   ├── models/                    # LLM model implementations
│   └── utils/                     # Utility modules
├── .env.example                  # Example environment variables
├── requirements.txt              # Python dependencies
└── README.md                     # This documentation
```

## Evaluation result

| Category | Models | Average (All) | Average (KO) | Average (EN) | Plan (KO) | Plan (EN) | Call Rejection (KO) | Call Rejection (EN) | Function Call (KO) | Function Call (EN) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Kakao | kanana-2-30b-a3b-thinking-2601 | 66.22 | 66.46 | 65.98 | 52.21 | 50.13 | 63.58 | 64.56 | 83.60 | 83.24 |
| K-AI | Solar-Open-100B | 67.05 | 65.09 | 69.01 | 42.01 | 47.84 | 74.83 | 78.86 | 78.43 | 80.32 |
| K-AI | K-EXAONE-236B-A23B | 58.30 | 59.67 | 56.92 | 35.39 | 36.92 | 60.14 | 54.43 | 83.47 | 79.42 |
| K-AI | HyperCLOVAX-SEED-Think-32B | 54.36 | 52.51 | 56.21 | 5.87 | 19.75 | 68.15 | 67.07 | 83.50 | 81.81 |
| K-AI | A.X-K1 | 69.94 | 71.98 | 67.90 | 56.07 | 40.87 | 78.11 | 79.81 | 81.77 | 83.01 |
| Closed Source | gemini-3-pro-preview | **84.12** | **84.24** | 84.00 | 82.02 | 82.08 | 86.84 | **87.02** | 83.87 | 82.90 |
| Closed Source | gemini-3-flash-preview | 83.54 | 83.00 | **84.08** | **83.83** | **83.59** | **87.48** | 85.77 | 77.69 | 82.88 |
| Closed Source | gpt-5.2-2025-12-11 | 76.05 | 77.08 | 75.01 | 70.68 | 70.10 | 76.94 | 74.60 | 83.63 | 80.34 |
| Open Source | GLM-4.7 | 83.17 | 84.12 | 82.22 | 76.43 | 75.78 | 86.35 | 83.45 | **89.59** | **87.42** |
| Open Source | Qwen3-235B-A22B-Instruct-2507 | 70.65 | 70.88 | 70.42 | 70.36 | 70.94 | 58.49 | 57.98 | 83.79 | 82.34 |
| Open Source | Qwen3-30B-A3B-Instruct-2507 | 66.08 | 66.11 | 66.04 | 72.37 | 73.16 | 44.16 | 44.00 | 81.79 | 80.96 |
| Open Source | Qwen2.5-32B-Instruct | 66.03 | 66.51 | 65.54 | 71.24 | 71.27 | 45.00 | 43.69 | 83.29 | 81.67 |
| Open Source | EXAONE-4.0-32B | 14.49 | 14.47 | 14.50 | 0.05 | 0.03 | 43.36 | 43.41 | 0.00 | 0.05 |
| Open Source | A.X-4.0 | 65.98 | 64.68 | 67.27 | 64.46 | 68.84 | 46.83 | 49.93 | 82.76 | 83.04 |

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- UV package manager (recommended) or pip
- API keys for the LLMs you want to test

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd OrchestrationBench
```

2. **Install dependencies:**
```bash
uv sync
# or with pip: pip install -e .
# Note: scipy is required for workflow evaluation
```

3. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env with your API keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
# Add other API keys as needed
```

### Basic Usage: Full Pipeline with `uv run evaluate`

The simplest way to run the full evaluation pipeline (scenario processing + evaluation) is using the `evaluate` CLI command with a YAML configuration file.

```bash
# Run evaluation with OpenAI models
uv run evaluate config/openai.yaml

# Run evaluation with Claude models
uv run evaluate config/claude.yaml

# Run evaluation with custom experiment ID
uv run evaluate config/openai.yaml --experiment-id my_experiment

# Run evaluation with custom output directory and verbose logging
uv run evaluate config/claude.yaml --results-dir ./my_results --verbose
```

**CLI Options:**
| Option | Short | Description |
|--------|-------|-------------|
| `config_path` | - | Path to the configuration YAML file (required) |
| `--experiment-id` | `-e` | Custom experiment ID. Defaults to timestamp if not provided |
| `--results-dir` | `-o` | Directory to save results. Defaults to `./results` |
| `--verbose` | `-v` | Enable verbose (DEBUG level) logging |

**Example Configuration Files:**
- `config/openai.yaml` - OpenAI models (GPT-4, GPT-4-mini, etc.)
- `config/claude.yaml` - Anthropic Claude models
- `config/gemini.yaml` - Google Gemini models
- `config/bedrock.yaml` - AWS Bedrock models
- `config/opensource_api.yaml` - Open-source models via external vLLM server
- `config/opensource_vllm.yaml` - Open-source models with auto-started vLLM

See the [Configuration](#️-configuration) section for details on creating custom configuration files.

---

### Advanced Usage: Running Individual Steps

For more granular control, you can run the scenario processing and evaluation steps separately.

#### Step 1: Scenario Processing (Data Generation)

Generate LLM interaction histories by running scenarios against the target model:

```bash
# Process all English scenarios with GPT-4
uv run python src/stepwise_scenario_processor.py \
  --model gpt-4.1-mini \
  --agent-cards data/EN/multiagent_cards \
  --data-path "data/EN/scenario_data/*.yaml" \
  --num-iter 3 \
  --output-dir data/results/step_wise_evaluation

# Process a specific scenario by ID
uv run python src/stepwise_scenario_processor.py \
  --model gpt-4.1-mini \
  --agent-cards data/EN/multiagent_cards \
  --data-path "data/EN/scenario_data/*.yaml" \
  --scenario-id 1 \
  --num-iter 3 \
  --output-dir data/results/step_wise_evaluation_TEST

# Process Korean scenarios
uv run python src/stepwise_scenario_processor.py \
  --model gpt-4.1-mini \
  --agent-cards data/KO/multiagent_cards \
  --data-path "data/KO/scenario_data/*.yaml" \
  --num-iter 3 \
  --output-dir data/results/step_wise_evaluation_KO
```

**Scenario Processor Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Model name defined in `config/base_config/multiagent_config.yaml` | Required |
| `--agent-cards` | Directory containing agent card JSON files (EN or KO) | Required |
| `--data-path` | Path pattern to scenario YAML files (supports glob patterns) | Required |
| `--scenario-id` | Process only a specific scenario by its ID | All scenarios |
| `--output-dir` | Directory where results will be saved | Required |
| `--max-scenarios` | Limit number of scenarios to process | Unlimited |
| `--num-iter` | Number of iterations per scenario | 1 |
| `--batch-size` | Number of concurrent agent executions | 5 |
| `--max-retries` | Maximum retry attempts for failed executions | 10 |

#### Step 2: Evaluation

Evaluate the generated results using the evaluation script:

```bash
# Evaluate results from a specific model
uv run python src/evaluation.py \
  --input data/results/step_wise_evaluation/gpt-4.1-mini \
  --agent-cards-path data/EN/multiagent_cards \
  --eval-config config/base_config/eval_config.yaml \
  --output evaluation_output.json \
  --sequential \
  --log-level INFO

# Evaluate with detailed LLM evaluation logs
uv run python src/evaluation.py \
  --input data/results/step_wise_evaluation/gpt-4.1-mini \
  --agent-cards-path data/EN/multiagent_cards \
  --save-llm-results \
  --llm-results-dir evaluation_logs

# Quick evaluation without LLM-based argument checking
uv run python src/evaluation.py \
  --input data/results/step_wise_evaluation/gpt-4.1-mini \
  --agent-cards-path data/EN/multiagent_cards \
  --skip-llm-eval
```

**Evaluation Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Path to input file or directory containing results | Required |
| `--agent-cards-path` | Directory containing agent card JSON files | Required |
| `--eval-config` | Path to evaluation configuration file | `config/base_config/eval_config.yaml` |
| `--output` | Output JSON file path | Auto-generated |
| `--sequential` | Process files one by one (recommended for stability) | False |
| `--skip-llm-eval` | Skip LLM-based argument evaluation (faster) | False |
| `--save-llm-results` | Save detailed LLM evaluation logs | False |
| `--llm-results-dir` | Directory for LLM evaluation logs | `llm_evaluation_logs` |
| `--start-over` | Delete temporary files and restart evaluation | False |
| `--pattern` | File pattern to match in directory mode | `*_out.json` |
| `--log-level` | Logging verbosity (DEBUG, INFO, WARNING, ERROR) | INFO |

## ⚙️ Configuration

### Evaluation Pipeline Configuration

The `uv run evaluate` command uses YAML configuration files with Hydra-style defaults. Config files inherit from `base_config/eval_config.yaml` and can override any settings.

> **Note:** For detailed configuration options and examples, refer to the individual config files in the `config/` directory (e.g., `openai.yaml`, `claude.yaml`, `gemini.yaml`, `bedrock.yaml`, `opensource_api.yaml`, `opensource_vllm.yaml`).

```yaml
# Hydra-style defaults (inherits judge model settings from base_config)
defaults:
  - base_config/eval_config
  - _self_

# Model configuration (target model to evaluate)
model:
  provider: openai          # Provider: openai, claude, gemini, bedrock, opensource
  model_alias: gpt-4.1-mini # Display name for results
  model: gpt-4.1-mini       # Actual model identifier
  base_url: https://api.openai.com/v1
  api_key: ${OPENAI_API_KEY} # Environment variable substitution
  temperature: 0.2
  # For opensource provider with local model:
  # model_path: /path/to/your/model

# Benchmark execution configuration
benchmark:
  temperature: 0.2          # Temperature for generation
  num_iter: 3               # Number of iterations per scenario
  batch_size: 50            # Concurrent executions
  max_retries: 10           # Retry attempts for failures

  # Optional: HuggingFace Hub upload
  # hf_hug_log_args:
  #   hub_results_org: your-org
  #   hub_repo_name: orchestration-bench-results
  #   push_results_to_hub: true
  #   public_repo: false

# Optional: Override judge model (defaults from base_config/eval_config)
# judge:
#   model:
#     provider: claude
#     model: claude-haiku-4-5-20251001
#     base_url: https://api.anthropic.com
#     api_key: ${ANTHROPIC_API_KEY}
#   generation_params:
#     temperature: 0.3
#     max_tokens: 12288
#     top_p: 1.0

# Optional: vLLM server configuration (only for opensource provider with model_path)
# vllm:
#   port: 15142
#   tensor_parallel_size: 1
#   gpu_memory_utilization: 0.95
#   max_model_len: null
#   reasoning_parser: null
#   tool_call_parser: null
#   extra_args:
#     - --trust-remote-code
```

**Supported Providers:**
| Provider | Description | Required Fields |
|----------|-------------|-----------------|
| `openai` | OpenAI API | `model`, `api_key`, `base_url` |
| `claude` | Anthropic Claude | `model`, `api_key`, `base_url` |
| `gemini` | Google Gemini | `model`, `api_key` |
| `bedrock` | AWS Bedrock | `model`, `aws_access_key_id`, `aws_secret_access_key`, `aws_region` (optional, default: `us-east-1`) |
| `opensource` | vLLM-served models | `model`, `model_url` or `model_path` |

**vLLM Configuration Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `port` | Port number for vLLM server | 15142 |
| `tensor_parallel_size` | Number of GPUs for tensor parallelism | 1 |
| `gpu_memory_utilization` | Fraction of GPU memory to use | 0.95 |
| `max_model_len` | Maximum model context length | Auto-detected |
| `reasoning_parser` | Parser for reasoning models | null |
| `tool_call_parser` | Parser for tool call outputs | null |
| `extra_args` | Additional vLLM command-line arguments | [] |

---

### Judge Model Configuration (`config/base_config/eval_config.yaml`)

Configure the judge model used for evaluation. This is the default configuration that can be overridden in provider-specific config files:

```yaml
# Judge model settings (used for evaluating model outputs)
judge:
  model:
    provider: openai
    base_url: https://api.openai.com/v1
    model: gpt-4.1
    api_key: ${OPENAI_API_KEY}
  generation_params:
    temperature: 0.3
    max_tokens: 12288
    top_p: 1.0

evaluation_params:
  agent_change_weight: 0.8
  status_change_weight: 0.2
```

**Overriding the Judge Model:**

You can override the judge model in any provider config file:

```yaml
# config/bedrock.yaml
defaults:
  - base_config/eval_config
  - _self_

model:
  provider: bedrock
  model: claude-sonnet-4
  ...

# Use Claude as judge instead of default
judge:
  model:
    provider: claude
    model: claude-haiku-4-5-20251001
    base_url: https://api.anthropic.com
    api_key: ${ANTHROPIC_API_KEY}
```

### Agent Cards

The system includes 17 specialized agents, each defined in JSON format:

#### English Agents (`data/EN/multiagent_cards/`)
- `calendar_agent.json` - Calendar and scheduling operations
- `weather_agent.json` - Weather information retrieval
- `search_agent.json` - Web search and information gathering  
- `place_agent.json` - Location and place information
- `transport_agent.json` - Transportation and routing
- `finance_agent.json` - Financial information and transactions
- `shopping_agent.json` - E-commerce and shopping tasks
- `entertainment_agent.json` - Movies, shows, and entertainment
- `travel_agent.json` - Travel planning and booking
- `message_agent.json` - Communication and messaging
- `news_agent.json` - News and current events
- `person_agent.json` - People and celebrity information
- `counsel_agent.json` - Counseling and advice
- `delivery_agent.json` - Package delivery and logistics
- `life_info_agent.json` - Daily life information
- `personal_banking_agent.json` - Personal banking operations
- `sports_agent.json` - Sports information and scores

#### Korean Agents (`data/KO/multiagent_cards/`)
Similar set of agents with Korean language support and localized capabilities.

## 📈 Results and Analysis

Results are saved in structured JSON format with comprehensive metrics for both workflow planning and tool execution evaluation.

### Key Metrics (Top-Level Summary)

| Metric | Formula | Description |
|--------|---------|-------------|
| **Average** | `(Call Rejection + FC + Plan) / 3` | Overall performance score |
| **Call Rejection Classification Accuracy** | `(reject_f1 + fc_f1) / 2` | Rejection/FC decision F1 average |
| **FC (Function Call)** | `(fn_name_f1 + key_f1 + value_f1) / 3` | Function call quality score |
| **Plan (Workflow)** | `workflow_evaluation_with_failure` | DAG-based planning score |

---

### Call Rejection Classification Accuracy

Measures the model's ability to decide **WHEN** to call tools vs **WHEN** to reject.

**Prediction types:**
- `FC`: Model makes a function call
- `Reject`: Model correctly decides not to call (with rejection type)
- `FAILED`: Model fails to generate a valid prediction

**Rejection cases:**
- `AWAITING_USER_INPUT`: Model should ask for more information
- `TOOL_CONSTRAINT_VIOLATION`: Model should recognize constraints prevent execution

```
Call Rejection = (reject_f1 + fc_f1) / 2
```

**Additional metrics:**
- `rejection_type_accuracy`: When model correctly rejects, how often is the rejection type correct?
- `total_rejection_type_mismatch`: Count of correct rejections with wrong type
- `total_failed_generation`: Count of failed predictions (tracked separately)

---

### FC (Function Call) Score

Measures function call quality when the model decides to call a tool.

```
FC = (function_name_f1 + arguments_key_f1 + arguments_value_f1) / 3

where:
  function_name_f1:   F1 score for correct tool/function name selection
  arguments_key_f1:   F1 score for correct parameter key matching
  arguments_value_f1: F1 score for correct parameter value matching
```

---

For detailed result format, see [`docs/result_format.md`](docs/result_format.md).

## 🤝 Contributing

Please submit issues or pull requests if you find problems with the benchmark or have suggestions for improvements.

## 📞 Contact

For questions or support, please open an issue on the repository.

## 📚 Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{OrchestrationBench2025,
  title={OrchestrationBench: LLM-driven Agentic Planning and Tool Use in Multi-Domain Scenarios},
  year={2025},
  url={[Repository URL]}
}
```

## License

This software is licensed under the Apache 2 license, quoted below.

Copyright 2025 Kakao Corp. http://www.kakaocorp.com

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this project except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
