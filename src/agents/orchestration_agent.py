"""
Orchestration Agent class for coordinating multiple agents.

This module provides the orchestration agent that coordinates workflows.
"""
import json
from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent
from src.agents.llm_agent import LLMAgent
from src.utils.type_utils import normalize_chat_response
from src.utils.config_loader import format_agent_cards_for_prompt
from copy import deepcopy
from loguru import logger
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionToolParam,
    ChatCompletionSystemMessageParam
)
from src.utils.model_factory import ModelFactory

class OrchestrationAgent(BaseAgent):
    """Agent for coordinating multiple agents in workflows."""
    
    def __init__(self, agent_id: str, agent_card_list: List[Dict[str, Any]], llm_config: Dict[str, Any], prompts: Dict[str, Any]):
        """Initialize the orchestration agent."""
        self.agent_id = agent_id  # Fixed: was agent_ids
        self.agent_card_list = agent_card_list
        self.llm_config = llm_config
        self.prompts = prompts
        self.managed_agents = {}  # Will be set by OrchestrationEngine
        self.model = None  # Initialize model attribute
        
        # Token tracking callback (set by OrchestrationEngine)
        self.token_callback = None
    
    def set_managed_agents(self, agents: Dict[str, Any]):
        """Set the agents this orchestration agent manages."""
        self.managed_agents = agents

    async def execute(self, msgs: list[ChatCompletionMessageParam], 
                      prev_workflow: Dict[str, Any] = None, 
                      system_info: str = "",
                      skip_summary: bool = True,
                      kwargs: Dict[str, Any] = None) -> str:
        """Execute orchestration agent - coordinate multiple agents."""
        if not self.model:
            await self.initialize()
        
        # For orchestration agent, we use the full workflow
        return await self.generate_workflow(msgs, prev_workflow, system_info, skip_summary, kwargs)

    async def summarization(self, msgs: list[ChatCompletionMessageParam]) -> list[ChatCompletionAssistantMessageParam]:
        """Summarize workflow execution."""
        if not self.model:
            await self.initialize()
        msgs_ = deepcopy(msgs)
        system_msg = self._get_system_prompt("summarize")
        msgs_.insert(0, system_msg)
        
        logger.debug(f"OrchestrationAgent: Starting summarization with {len(msgs)} messages")
        
        # Generate summary
        response = await self.model.generate_chat_response(
            messages=msgs_,
        )
        response = normalize_chat_response(response)
        
        logger.debug(f"OrchestrationAgent: Summarization completed, response type: {type(response)}")
        return response

    async def classify_intent(self, msgs: list[ChatCompletionMessageParam], 
                                system_info: str = "", 
                                prev_workflow: Dict[str, Any] = None, 
                                kwargs: Dict[str, Any] = None) -> str:
        """Execute orchestration agent - coordinate multiple agents."""
        if not self.model:
            await self.initialize()
        
        # For orchestration agent, we use the full workflow
        return await self._classify_intent(msgs, system_info, prev_workflow, kwargs)

    async def _classify_intent(self, msgs: list[ChatCompletionMessageParam],
                              system_info: str = "", 
                              prev_workflow: Dict[str, Any] = None, 
                              kwargs: Dict[str, Any] = None) -> dict:
        msgs_ = deepcopy(msgs)
        prompt_template = self._get_system_prompt("classify", as_message=False)
        prompt_template = prompt_template.replace("%%system_info%%", system_info)
        prompt_template = prompt_template.replace("%%workflows%%", json.dumps(prev_workflow, ensure_ascii=False) if prev_workflow is not None and prev_workflow != {} else "NONE")
        system_msg = ChatCompletionSystemMessageParam(
                    role="system",
                    content=prompt_template
                )
        msgs_.insert(0, system_msg)
        # Generate workflow
        response = await self.model.generate_chat_response(
            messages=msgs_,
        )
        response = normalize_chat_response(response)
        return response
    

    async def generate_workflow(self, msgs: list[ChatCompletionMessageParam], 
                                prev_workflow: Dict[str, Any] = None,
                                system_info: str = "", 
                                skip_summary: bool = True, 
                                kwargs: dict = None) -> dict:
        """Generate workflow JSON based on input messages."""
        msgs_ = deepcopy(msgs)
        prompt_template = self._get_system_prompt("general", as_message=False)
        # Replace placeholders - convert agent cards to formatted string
        agent_cards_text = format_agent_cards_for_prompt(self.agent_card_list) if isinstance(self.agent_card_list, dict) else ""
        prompt = prompt_template.replace("%%agent_cards%%", agent_cards_text)
        prompt = prompt.replace("%%system_info%%", system_info)
        prompt = prompt.replace("%%workflows%%", json.dumps(prev_workflow, ensure_ascii=False) if prev_workflow is not None and prev_workflow != {} else "NONE")

        # Debug: Log the conditions
        logger.debug(f"OrchestrationAgent: len(msgs)={len(msgs)}, skip_summary={skip_summary}")
        
        history_text = ""
        for i, msg in enumerate(msgs):
            role = msg.get("role", "")
            content = msg.get("content", "")
            history_text += f"[Message {i+1}] {role}: {content}\n\n"
        prompt = prompt.replace("%%history%%", history_text)
        logger.debug(f"OrchestrationAgent: Added {len(msgs)} messages to history")

        system_msg = ChatCompletionSystemMessageParam(
                    role="system",
                    content=prompt
                )
        
        # When skip_summary=True, use a simple user message asking for workflow generation
        # since all context is already in the system prompt
        if skip_summary and len(msgs) > 0:
            last_user_msg = None
            for msg in reversed(msgs):
                if msg.get("role") == "user":
                    last_user_msg = msg
                    break
            
            if last_user_msg is None:
                simple_msgs = [
                    system_msg,
                    {"role": "user", "content": "Based on the conversation history above, please generate the appropriate workflow."}
                ]
            else:
                simple_msgs = [system_msg] + msgs_
        else:
            msgs_.insert(0, system_msg)
            simple_msgs = msgs_
            
        # Generate workflow
        response = await self.model.generate_chat_response(
            messages=simple_msgs,
        )
        response = normalize_chat_response(response)
        # Debug: Check response type and structure
        logger.debug(f"Model response type: {type(response)}")
        logger.debug(f"Model response content: {response}")
        
        return response

    def _get_system_prompt(self, node: str, as_message: bool = True) -> ChatCompletionSystemMessageParam:
        prompt = self.prompts.get(node).get("prompt")
        if as_message:
            return ChatCompletionSystemMessageParam(role="system", content=prompt)
        return prompt

    async def _generate_summary(self, msgs: list[ChatCompletionMessageParam]) -> str:
        """Generate summary of conversation history."""
        msgs_ = deepcopy(msgs)
        system_msg = self._get_system_prompt("history_summarization")

        msgs_.insert(0, system_msg)
        # Generate summary
        response = await self.model.generate_chat_response(
            messages=msgs_,
        )
        # Ensure response is a string
        if isinstance(response, dict):
            # If response is a dict, extract content
            if "content" in response:
                return str(response["content"])
            elif "message" in response and "content" in response["message"]:
                return str(response["message"]["content"])
            else:
                return str(response)
        return str(response)

    async def initialize(self):
        """Initialize the orchestration agent."""
        if not self.model:
            self.model = await self._setup_model_connection()
        logger.debug(f"OrchestrationAgent {self.agent_id} model initialized")
    
    async def _setup_model_connection(self) -> Any:
        """Set up LLM model connection based on configuration."""
        try:
            model = await ModelFactory.create_model(
                llm_config=self.llm_config,
                token_callback=self.token_callback
            )
            logger.debug(f"Successfully initialized model for OrchestrationAgent {self.agent_id}")
            return model
        except Exception as e:
            logger.error(f"Failed to initialize model for OrchestrationAgent {self.agent_id}: {e}")
            raise
    
    async def select_tools(self, query: str, system_prompt: str = "") -> List[Dict[str, Any]]:
        """Select appropriate tools for the query. OrchestrationAgent doesn't use tools."""
        return []  # Orchestrator doesn't use tools directly
    
    async def execute_tools(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute tool calls. OrchestrationAgent doesn't use tools."""
        return []  # Orchestrator doesn't execute tools directly
    
    async def generate_response(self, query: str, tool_results: List[Dict[str, Any]], system_prompt: str = "") -> str:
        """Generate final response. For orchestrator, this delegates to generate_workflow."""
        # Convert query to message format
        msgs = [{"role": "user", "content": query}]
        return await self.generate_workflow(msgs)
    
    async def get_scenario(self, data: Dict[str, Any]):
        """
        Receive scenario data for scenario-based operation.
        
        Args:
            data: Dictionary containing scenario data from YAML file
        """
        # OrchestrationAgent can also receive scenario data for context
        logger.debug(f"🎭 OrchestrationAgent {self.agent_id} received scenario data with {len(data.get('steps', []))} steps")
