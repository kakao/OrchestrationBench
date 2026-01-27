"""
LLM Agent class with full implementation.

This module provides the LLM agent class with complete functionality.
"""

from typing import Dict, Any, List, Optional, Tuple
import json
from copy import deepcopy
from rich.console import Console
from src.agents.base_agent import BaseAgent
from src.utils.model_factory import ModelFactory

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionToolParam,
    ChatCompletionSystemMessageParam
)
from loguru import logger
from src.utils.type_utils import is_tool_message, get_content, normalize_chat_response
from src.utils.shared_types import CompletionRequest
from src.utils.toolcall_fetcher import ToolResultFetcher
from src.utils.type_utils import is_tool_message, is_assistant_message, is_user_message
import traceback

class LLMAgent(BaseAgent):
    """LLM Agent class with full implementation of common functionality."""
    
    def __init__(self, agent_id: str, 
                        agent_card: Dict[str, Any], 
                        llm_config: Dict[str, Any], 
                        prompts: Dict[str, Any], 
                        scenario_based: bool = True
                 ):
        super().__init__(agent_id, agent_card, llm_config, prompts, scenario_based=scenario_based)
        
        # Build tools, prompt, and required fields
        self.tools = agent_card.get("tools", [])
        self.required_fields_map = self._build_required_fields_map()
        
        # Parse multi-step prompts if available
        if "prompt" in agent_card:
            self.agent_prompt = agent_card["prompt"] 
        else:
            self.agent_prompt = prompts.get("agent_default", {}).get("prompt", "")
        
        # Token tracking callback (set by OrchestrationEngine)
        self.token_callback = None
        super().__init__(agent_id, agent_card, llm_config, prompts)
        self.model = None
        logger.debug(f"🤖 Initialized LLM agent: {agent_id}")
        self.scenario_based = scenario_based
        if self.scenario_based:
            self.external_scenario = None  # Placeholder for scenario data
            self.required_fields_map = self._build_required_fields_map()
        else:
            self.required_fields_map = {}
        logger.debug(f"🤖 Initialized LLM agent: {agent_id}")
    
    def _build_required_fields_map(self) -> Dict[str, List[str]]:
        """도구 스키마에서 required fields 매핑 생성"""
        if not hasattr(self, 'tools') or not self.tools:
            return {}
        
        return {
            tool['name']: tool.get('parameters', {}).get('required', [])
            for tool in self.tools
            if 'name' in tool
        }
    
    def get_required_fields(self, tool_name: str) -> List[str]:
        """특정 도구의 required fields 반환"""
        return self.required_fields_map.get(tool_name, [])

    async def initialize(self):
        """Initialize the agent's LLM model."""
        if not self.model:
            try:
                self.model = await self._setup_model_connection()
                if self.model is None:
                    raise RuntimeError(f"Failed to initialize model for agent {self.agent_id}")
            except Exception as e:
                logger.error(f"Failed to initialize model for agent {self.agent_id}: {e}")
                raise
        logger.debug(f"Agent {self.agent_id} model initialized")
        # logger.debug(f"Tools {self.tools} Initialized for agent {self.agent_id}")

    async def _setup_model_connection(self) -> Any:
        """Set up LLM model connection based on configuration."""
        try:
            model = await ModelFactory.create_model(
                llm_config=self.llm_config,
                token_callback=self.token_callback
            )
            logger.debug(f"Successfully initialized model for agent {self.agent_id}")
            return model
        except Exception as e:
            logger.error(f"Failed to initialize model for agent {self.agent_id}: {e}")
            raise
    
    def tidy_up(self, msgs: list[ChatCompletionMessageParam]) -> list[ChatCompletionMessageParam]:
        try:
            logger.debug(f"🔧 Agent {self.agent_id} - tidy_up input type: {type(msgs)}, length: {len(msgs) if isinstance(msgs, list) else 'not a list'}")
            
            msgs_ = deepcopy(msgs)
            for i, msg in enumerate(msgs_):
                logger.debug(f"🔧 Agent {self.agent_id} - processing msg {i}: type={type(msg)}, keys={list(msg.keys()) if isinstance(msg, dict) else 'not a dict'}")
                
                if not is_tool_message(msg):
                    continue
                    
                try:
                    content = get_content(msg)
                    parsed = json.loads(content)
                    for p in parsed.get("products", []):
                        keys_deleted = [k for k in p.keys() if k.startswith("canvas_")]
                        for k in keys_deleted:
                            del p[k]
                    msg["content"] = json.dumps(parsed, ensure_ascii=False)
                except (KeyError, ValueError) as e:
                    logger.debug(f"🔧 Agent {self.agent_id} - msg {i} processing error: {e}")
                    continue
                    
            return msgs_
        except Exception as e:
            logger.error(f"❌ Agent {self.agent_id} - tidy_up error: {e}")
            logger.error(f"❌ Agent {self.agent_id} - msgs type: {type(msgs)}")
            logger.error(f"❌ Agent {self.agent_id} - msgs content: {msgs}")
            raise

    def _get_system_prompt(self, node: str, system_info: str = "") -> ChatCompletionSystemMessageParam:
        prompt = self.prompts.get(node).get("prompt")
        if system_info:
            prompt = f"SYSTEM_INFO:\n{system_info}\n\n{prompt}"
        return ChatCompletionSystemMessageParam(role="system", content=prompt)

    async def get_scenario(self, data: Dict[str, Any]):
        """
        Receive scenario data for scenario-based operation.
        
        Args:
            data: Dictionary containing scenario data from YAML file
        """
        if self.scenario_based:
            self.external_scenario = data
            logger.debug(f"🎭 Agent {self.agent_id} received scenario data with {len(data.get('steps', []))} steps")
            
            # Extract relevant information for this agent
            agent_steps = [
                step for step in data.get('steps', [])
                if step.get('agent') == self.agent_id or step.get('agent_type') == self.agent_id
            ]
            
            if agent_steps:
                logger.debug(f"🎯 Found {len(agent_steps)} relevant steps for agent {self.agent_id}")
        else:
            logger.debug(f"🚫 Agent {self.agent_id} is not scenario-based, ignoring scenario data")
    
    def get_scenario_data(self) -> Optional[Dict[str, Any]]:
        """
        Get the current scenario data.
        
        Returns:
            Dictionary containing scenario data, or None if not set
        """
        return self.external_scenario if self.scenario_based else None
    
    def get_agent_specific_steps(self) -> List[Dict[str, Any]]:
        """
        Get scenario steps that are specific to this agent.
        
        Returns:
            List of steps where this agent is involved
        """
        if not self.external_scenario:
            return []
        
        return [
            step for step in self.external_scenario.get('steps', [])
            if step.get('agent') == self.agent_id or step.get('agent_type') == self.agent_id
        ]
    async def execute(self, req: CompletionRequest, system_info: str = "") -> str:
        """Execute a task using the agent."""
        try:
            # Always check and re-initialize model if needed
            if self.model is None:
                logger.debug(f"Model is None for agent {self.agent_id}, re-initializing...")
                await self.initialize()
            
            # Double check after initialization
            if self.model is None:
                logger.error(f"Model still None after initialization for agent {self.agent_id}")
                raise RuntimeError(f"Model for agent {self.agent_id} is not initialized")
            
            # Verify model has required method
            if not hasattr(self.model, 'generate_chat_response'):
                logger.error(f"Model for agent {self.agent_id} missing generate_chat_response method")
                raise RuntimeError(f"Model for agent {self.agent_id} is invalid")
            
            return await self.chat_completions(req=req, system_info=system_info)
        except Exception as e:
            logger.error(f"Error in agent {self.agent_id} execution: {e}")
            raise
    
    async def chat_completions(
        self, req: CompletionRequest, system_info: str = ""
    ) -> List[ChatCompletionMessageParam]:
        """Handle chat completions with tool selection and execution."""
        # Extract messages from CompletionRequest
        try:
            logger.debug(f"🔧 Agent {self.agent_id} - req.agent_messages keys: {list(req.agent_messages.keys())}")
            logger.debug(f"🔧 Agent {self.agent_id} - req.agent_messages type: {type(req.agent_messages)}")
            
            if self.agent_id in req.agent_messages:
                raw_msgs = req.agent_messages[self.agent_id]
                logger.debug(f"🔧 Agent {self.agent_id} - raw_msgs type: {type(raw_msgs)}, content: {raw_msgs}")
                msgs = self.tidy_up(raw_msgs)
            else:
                # Fallback to first available messages or empty list
                agent_msgs = list(req.agent_messages.values())
                raw_msgs = agent_msgs[0] if agent_msgs else []
                logger.debug(f"🔧 Agent {self.agent_id} - fallback raw_msgs type: {type(raw_msgs)}, content: {raw_msgs}")
                msgs = self.tidy_up(raw_msgs)
        except Exception as e:
            logger.error(f"❌ Agent {self.agent_id} - Error extracting messages: {e}")
            logger.error(f"❌ Agent {self.agent_id} - req type: {type(req)}")
            logger.error(f"❌ Agent {self.agent_id} - req content: {req}")
            raise
        
        turn_resps: List[ChatCompletionMessageParam] = []
        
        while True:
            new_resps = await self._dispatch(msgs, system_info=system_info)
            if not new_resps:
                break
            msgs.extend(new_resps)
            turn_resps.extend(new_resps)
        return turn_resps
    
    async def _dispatch(
        self, msgs: List[ChatCompletionMessageParam], system_info: str = ""
    ) -> List[ChatCompletionMessageParam]:
        """Dispatch messages to appropriate handler."""
        try:
            logger.debug(f"🔧 Agent {self.agent_id} - _dispatch called with {len(msgs)} messages")
            
            if not msgs:
                logger.debug("No messages to dispatch")
                return []
                
            last_msg = msgs[-1]
            logger.debug(f"🔧 Agent {self.agent_id} - last_msg type: {type(last_msg)}, keys: {list(last_msg.keys()) if isinstance(last_msg, dict) else 'not a dict'}")
            
            # Safety check for message format
            if not isinstance(last_msg, dict):
                logger.error(f"Invalid message format: {type(last_msg)}, content: {last_msg}")
                return []
                
            if "role" not in last_msg:
                logger.error(f"Message missing 'role' field: {last_msg}")
                return []        
        except Exception as e:
            logger.error(f"❌ Agent {self.agent_id} - _dispatch error: {e}")
            logger.error(f"❌ Agent {self.agent_id} - msgs type: {type(msgs)}")
            logger.error(f"❌ Agent {self.agent_id} - msgs content: {msgs}")
            raise
        resps = []

        if is_user_message(last_msg):
            resps = [await self._complete_general(msgs, system_info=system_info)]
        elif is_assistant_message(last_msg) and last_msg.get("tool_calls"):
            resps = await self._call_tools(msgs)
        elif last_msg.get("role") == "tool":
            # Tool response received - call LLM again to process tool results
            resps = [await self._complete_general(msgs, system_info=system_info)]
        elif last_msg.get("role") == "system":
            # If last message is system, treat it as a user query by creating a user message
            user_query = last_msg.get("content", "")
            # Remove the system message and replace with user message
            msgs_with_user = msgs[:-1] + [{"role": "user", "content": user_query}]
            resps = [await self._complete_general(msgs_with_user, system_info=system_info)]
        return resps
    
    async def _complete_general(
        self, msgs: list[ChatCompletionMessageParam], system_info: str = ""
    ) -> ChatCompletionMessageParam:
        msgs_ = deepcopy(msgs)
        
        # Check if there's already a system message
        has_system_msg = msgs_ and msgs_[0].get("role") == "system"
        
        if not has_system_msg:
            # Only insert system prompt if there isn't one already
            msgs_.insert(0, self._get_system_prompt("select_tools", system_info=system_info))
        
        kwargs = {
            "tools": self.tools,   
        }

        logger.debug(f"🔧 Agent {self.agent_id} - calling model.generate_chat_response")
        logger.debug(f"🔧 Agent {self.agent_id} - msgs_ length: {len(msgs_)}, has_system: {has_system_msg}")
        resp = await self.model.generate_chat_response(msgs_, **kwargs)
        if not resp:
            logger.error(f"❌ Agent {self.agent_id} - model response is empty")
            return {"role": "assistant", "content": "No response from model"}
        resp = normalize_chat_response(resp)
        logger.debug(f"🔧 Agent {self.agent_id} - model response type: {type(resp)}")
        logger.debug(f"🔧 Agent {self.agent_id} - model response content: {resp}")
        
        # Simply return the response - let the workflow handler manage tool calls
        return resp

    async def _call_tools(self, msgs: List[ChatCompletionMessageParam]
    ) -> List[ChatCompletionMessageParam]:
        """Handle tool calls with scenario support."""
        
        last_msg = msgs[-1]
        if not (is_assistant_message(last_msg) and last_msg.get("tool_calls")):
            return []
        
        tool_responses = []
        for tool_call in last_msg["tool_calls"]:
            tool_name = tool_call["function"]["name"]
            arguments_string = tool_call["function"]["arguments"]
            tool_call_id = tool_call["id"]
            
            try:
                if self.scenario_based:
                    # Scenario based tool call
                    result = await self._handle_scenario_tool_call(
                        tool_name, arguments_string, 
                    )
                else:
                    # 일반 도구 호출
                    result = await self._handle_regular_tool_call(
                        tool_name, arguments_string
                    )
            except Exception as e:
                logger.error(f"Error calling tool {tool_name}: {e}")
                result = {"error": f"Tool execution failed: {str(e)}"}
            
            # 도구 응답 메시지 생성
            tool_response: ChatCompletionToolMessageParam = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(result, ensure_ascii=False)
            }
            tool_responses.append(tool_response)
        return tool_responses

    async def _handle_scenario_tool_call(
        self, tool_name: str, tool_call_str: str
    ) -> Dict[str, Any]:
        """Handle tool calls using scenario data."""
        logger.debug(f"************************ Handling scenario tool call: {tool_name} with args: {tool_call_str}")
        try:
            # scenario data에서 steps 부분만 추출하여 전달
            scenario_steps = self.external_scenario.get('steps', []) if isinstance(self.external_scenario, dict) else []
            fetcher = ToolResultFetcher(scenario_steps)
            required_fields = self.get_required_fields(tool_name)
            result = fetcher.get_tool_result(tool_name, tool_call_str, required_fields)
            
            logger.debug(f"Scenario tool call result for {tool_name}: {result}")
            return result

        except Exception as e:
            logger.error(f"Error in scenario tool call {tool_name}: {e}")
            return {"error": f"Scenario tool execution failed: {str(e)}"}

    async def _handle_regular_tool_call(
        self, tool_name: str, arguments_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle regular tool calls."""
        # Write the actual tool execution logic here
        # For example, database queries, API calls, file operations, etc.
        
        if tool_name == "search_database":
            # 예시: 데이터베이스 검색
            return await self._search_database(arguments_dict)
        elif tool_name == "call_api":
            # 예시: API 호출
            return await self._call_external_api(arguments_dict)
        else:
            # 기본 도구 처리
            return {"message": f"Tool {tool_name} executed with args: {arguments_dict}"}

