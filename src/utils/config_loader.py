"""
Configuration loader for YAML files and agent cards.
"""

import json
import yaml
import os
import re
from pathlib import Path
from typing import Dict, Any
from loguru import logger
from rich.console import Console

console = Console()


def load_dotenv(dotenv_path: Path = None) -> None:
    """Load environment variables from .env file."""
    if dotenv_path is None:
        # Look for .env file in project root
        project_root = Path(__file__).parent.parent.parent
        dotenv_path = project_root / ".env"
    
    if not dotenv_path.exists():
        return
    
    with open(dotenv_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and not os.getenv(key):  # Don't override existing env vars
                    os.environ[key] = value


def substitute_env_vars(content: str) -> str:
    """Substitute environment variables in content."""
    def replace_env_var(match):
        var_name = match.group(1)
        return os.getenv(var_name, f"${{{var_name}}}")  # Keep original if not found
    
    return re.sub(r'\$\{([^}]+)\}', replace_env_var, content)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file with environment variable substitution and overrides."""
    try:
        # Load .env file first
        load_dotenv()

        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Substitute environment variables
        content = substitute_env_vars(content)

        config = yaml.safe_load(content)

        # Apply environment variable overrides for model config
        config = apply_env_overrides(config)

        return config
    except Exception as e:
        logger.debug(f"[red]Error loading config: {e}[/red]")
        raise


def apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply environment variable overrides to config.

    Environment variables:
    - ORCHESTRATION_BENCH_MODEL_ALIAS: model alias (key in llms dict)
    - ORCHESTRATION_BENCH_MODEL: actual model name
    - ORCHESTRATION_BENCH_BASE_URL: API base URL
    - ORCHESTRATION_BENCH_API_KEY: API key
    - ORCHESTRATION_BENCH_PROVIDER: provider type (openai, claude, bedrock, etc.)
    - ORCHESTRATION_BENCH_TEMPERATURE: temperature setting
    - AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION: for Bedrock
    """
    model_alias = os.getenv("ORCHESTRATION_BENCH_MODEL_ALIAS")
    if not model_alias:
        return config

    # Get override values from environment
    model = os.getenv("ORCHESTRATION_BENCH_MODEL")
    base_url = os.getenv("ORCHESTRATION_BENCH_BASE_URL")
    api_key = os.getenv("ORCHESTRATION_BENCH_API_KEY")
    provider = os.getenv("ORCHESTRATION_BENCH_PROVIDER")
    temperature = os.getenv("ORCHESTRATION_BENCH_TEMPERATURE")

    # Bedrock specific
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION", os.getenv("ORCHESTRATION_BENCH_AWS_REGION"))

    # Build model config
    model_config = {}

    if provider == "bedrock":
        if aws_access_key_id:
            model_config["AWS_ACCESS_KEY_ID"] = aws_access_key_id
        if aws_secret_access_key:
            model_config["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
        if aws_region:
            model_config["AWS_REGION"] = aws_region
        if model:
            model_config["model"] = model
    else:
        if base_url:
            model_config["base_url"] = base_url
        if model:
            model_config["model"] = model
        if api_key:
            model_config["api_key"] = api_key
        if provider:
            model_config["provider"] = provider
        if temperature:
            model_config["temperature"] = float(temperature)

    # Only apply if we have overrides
    if model_config:
        if "llms" not in config:
            config["llms"] = {}
        config["llms"][model_alias] = model_config
        logger.debug(f"Applied env overrides for model: {model_alias}")

    return config


def load_agent_cards(cards_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load all agent cards from directory."""
    cards = {}
    if not cards_dir.exists():
        logger.debug(f"[yellow]Agent cards directory not found: {cards_dir}[/yellow]")
        return cards
    
    for card_file in cards_dir.glob("*.json"):
        try:
            with open(card_file, 'r', encoding='utf-8') as f:
                card_data = json.load(f)
                agent_id = card_data.get("agent_card", {}).get("agent_id")
                if agent_id:
                    cards[agent_id] = {"agent_card": card_data["agent_card"], 
                                       "tools": card_data.get("tools", [])}
                    logger.debug(f"✅ Loaded agent card: {agent_id}")
        except Exception as e:
            logger.debug(f"[red]Error loading agent card {card_file}: {e}[/red]")
    
    return cards


def format_agent_cards_for_prompt(cards: Dict[str, Dict[str, Any]]) -> str:
    """Format agent cards for inclusion in prompts."""
    formatted = []
    for agent_id, card in cards.items():
        # Skip empty or invalid cards
        if not card or not isinstance(card, dict):
            continue
        card = card.get("agent_card", {})
        # Safely get name and description
        name = card.get('name', agent_id)
        description = card.get('description', 'No description available')
        
        formatted.append(f"**{name} ({agent_id})**")
        formatted.append(f"- {description}")
        
        if 'skills' in card and card['skills']:
            formatted.append("- Skills:")
            for skill in card['skills']:  # Show first 3 skills
                skill_id = skill.get('id', 'Unknown skill')
                skill_desc = skill.get('description', 'No description')
                formatted.append(f"  - {skill_id}: {skill_desc}")
        
        formatted.append("")
    
    return "\n".join(formatted)
