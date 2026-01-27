"""
Orchestration Engine for multi-agent workflow execution.

This module implements a simplified orchestration engine that only includes
functions used by stepwise_scenario_processor and step_history_generator.
"""

from typing import Dict, Any
from pathlib import Path
from src.utils.config_loader import load_config, load_agent_cards
from src.utils.token_tracker import TokenTracker

from src.agents.orchestration_agent import OrchestrationAgent
from src.agents.llm_agent import LLMAgent

# Simple logging function to replace loguru
def debug_log(message: str):
    print(f"DEBUG: {message}")


class OrchestrationEngine:
    """Simplified orchestration engine that coordinates agents for evaluation."""
    
    def __init__(self, config_file: Path, agent_cards_dir: Path, model_type: str = "openai"):
        """
        Initialize the orchestration engine.
        
        Args:
            config_file: Path to the YAML configuration file
            agent_cards_dir: Path to directory containing agent card JSON files
        """
        self.config_file = config_file
        self.agent_cards_dir = agent_cards_dir
        self.model_type = model_type
        
        # Configuration and agent data
        self.config_data = {}
        self.agent_cards = {}
        self.agents = {}  # {agent_id: LLMAgent}
        
        # Token usage tracking
        self.token_tracker = TokenTracker()
        
        # Initialize
        self._load_configuration()
        self._load_agent_cards()
        self._initialize_agents()
        
    def _load_configuration(self):
        """Load configuration from YAML file."""
        self.config_data = load_config(self.config_file)
    
    def _load_agent_cards(self):
        """Load agent cards and extract tools."""
        self.agent_cards = load_agent_cards(self.agent_cards_dir)
        debug_log(f"✅ Loaded {len(self.agent_cards)} agent cards")
    
    def _initialize_agents(self):
        """Initialize Agent objects for each agent."""
        agent_info = self.config_data.get("agent_info", {})
        llms = self.config_data.get("llms", {})
        prompts = self.config_data.get("prompts", {})
        
        # Create agent objects for each agent card
        debug_log(f"Initializing agents with {len(self.agent_cards)} cards")
        for agent_id, agent_card in self.agent_cards.items():
            llm_config = llms.get(self.model_type, {})
            if llm_config:
                # Create LLM agent
                agent = LLMAgent(
                    agent_id=agent_id,
                    agent_card=agent_card,
                    llm_config=llm_config,
                    prompts=prompts,
                    scenario_based=True
                )
                # Set token tracking callback for this agent
                agent.token_callback = self.token_tracker.create_callback(
                    agent_id=agent_id,
                    operation="agent_execution"
                )
                self.agents[agent_id] = agent
                debug_log(f"📋 Created agent: {agent_id} with {len(agent_card.get('tools', []))} tools")

        # Create orchestrator agent
        orchestrator = OrchestrationAgent(
            agent_id="orchestrator",
            agent_card_list=self.agent_cards,
            llm_config=llm_config,
            prompts=prompts
        )
        # Set token tracking callback for orchestrator
        orchestrator.token_callback = self.token_tracker.create_callback(
            agent_id="orchestrator",
            operation="workflow_generation"
        )
        self.agents["orchestrator"] = orchestrator
        
        if orchestrator:
            orchestrator.set_managed_agents(self.agents)

    def get_token_usage_summary(self) -> Dict[str, Any]:
        """
        Get a summary of token usage across all agents.
        
        Returns:
            Dictionary containing token usage statistics
        """
        summary = self.token_tracker.get_total_summary()
        return {
            "total_calls": summary.total_calls,
            "total_input_tokens": summary.total_input_tokens,
            "total_output_tokens": summary.total_output_tokens,
            "total_tokens": summary.total_tokens,
            "total_cost": summary.total_cost
        }
