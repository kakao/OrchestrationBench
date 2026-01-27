"""
Base Agent class for all agents.

This module provides the abstract base agent class with interface definitions.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from rich.console import Console
from loguru import logger
console = Console()


class BaseAgent(ABC):
    """Abstract base class for all agents with common interface."""
    
    def __init__(self, agent_id: str, agent_card: Dict[str, Any], llm_config: Dict[str, Any], prompts: Dict[str, Any], scenario_based: bool = True):
        """
        Initialize the base agent.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_card: Agent card configuration
            llm_config: LLM configuration for this agent
            prompts: Prompts configuration
            scenario_based: Whether the agent operates in a scenario-based context
        """
        self.agent_id = agent_id
        self.agent_card = agent_card.get("agent_card", [])
        self.llm_config = llm_config
        self.prompts = prompts
        self.scenario_based = scenario_based
        self.tools = [{"type": "function", "function": tool} for tool in agent_card.get("tools", [])]
        
        logger.debug(f"[blue]🤖 Initialized base agent: {agent_id}[/blue]")
    
    @abstractmethod
    async def initialize(self):
        """Initialize the agent. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def execute(self, query: str, args: Dict[str, Any] = None) -> str:
        """
        Execute the agent with the given query. Must be implemented by subclasses.
        
        Args:
            query: The query to execute
            args: Additional arguments
            
        Returns:
            Agent response
        """
        pass