"""
Simple History Generator

This module processes EvaluationStep data and generates conversation history
following the specific workflow outlined in the user requirements.
"""

import asyncio
import re
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from rich.console import Console
from rich.panel import Panel
import traceback
from src.orchestration_engine import OrchestrationEngine
from openai.types.chat import ChatCompletionMessageParam
from loguru import logger
import traceback
import secrets
import base64

console = Console()


def generate_custom_hash(prefix="call_", length=22):
    # Generate random bytes (length 16 gives ~22 base64 chars)
    random_bytes = secrets.token_bytes(16)
    # URL-safe base64 encode (remove padding)
    hash_str = base64.urlsafe_b64encode(random_bytes).decode('utf-8').rstrip('=')
    # Truncate or pad to desired length
    hash_str = (hash_str[:length]).ljust(length, 'A')
    return f"{prefix}{hash_str}"

@dataclass
class EvaluationStep:
    """Represents a single evaluation step"""
    workflow_id: Optional[str]
    step_id: int
    agent: str
    agent_type: str
    data: Dict[str, Any]

class SimpleHistoryGenerator:
    """Generates conversation history from EvaluationStep data"""
    
    def __init__(self, config_file: Path, agent_cards_dir: Path, model_type: str = "openai", batch_size: int = 5, max_retries: int = 1):
        """
        Initialize the simple history generator.
        
        Args:
            config_file: Path to the YAML configuration file
            agent_cards_dir: Path to directory containing agent card JSON files
            model_type: Type of model to use (openai, gemini, etc.)
            batch_size: Maximum number of concurrent agent executions per batch
            max_retries: Maximum number of retry attempts for failed agent executions
        """
        self.config_file = config_file
        self.agent_cards_dir = agent_cards_dir
        self.batch_size = batch_size
        self.max_retries = max_retries

        # Initialize components
        self.orchestration_engine = None
        self.orchestration_agent = None
        self.model_type = model_type

        
    async def initialize(self):
        """Initialize all components"""
        try:
            # Initialize orchestration engine
            self.orchestration_engine = OrchestrationEngine(
                config_file=self.config_file,
                agent_cards_dir=self.agent_cards_dir,
                model_type=self.model_type
            )
            
            # Get orchestration agent
            self.orchestration_agent = self.orchestration_engine.agents.get("orchestrator")
            if self.orchestration_agent:
                await self.orchestration_agent.initialize()
            if not self.orchestration_agent:
                raise ValueError("Orchestrator not found in agents")

        except Exception as e:
            logger.debug(f"❌ Initialization failed: {e}", style="red")
            raise

    async def close(self):
        """Close all model sessions to prevent resource leaks."""
        try:
            if self.orchestration_engine:
                for agent_id, agent in self.orchestration_engine.agents.items():
                    if hasattr(agent, 'model') and agent.model and hasattr(agent.model, 'close'):
                        await agent.model.close()
                        logger.debug(f"✅ Closed model session for agent: {agent_id}")
        except Exception as e:
            logger.debug(f"⚠️ Error closing model sessions: {e}")

    def parse_workflow(self, workflow_content: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Parse workflow content to extract workflow_id -> steps mapping.
        
        Args:
            workflow_content: YAML-like workflow content
            
        Returns:
            Dictionary mapping workflow_id to list of steps with agent_id and refined_query
        """
        all_workflows = {}
        
        try:
            # Parse YAML content
            workflow_data = yaml.safe_load(workflow_content)
            
            if isinstance(workflow_data, dict):
                for workflow_id, workflow_info in workflow_data.items():
                    if isinstance(workflow_info, dict) and "steps" in workflow_info:
                        steps = workflow_info["steps"]
                        workflow_steps = []
                        if isinstance(steps, list):
                            for step in steps:
                                if isinstance(step, dict):
                                    agent_id = step.get("name", "")
                                    refined_query = step.get("refined_query", "")
                                    if agent_id and refined_query:
                                        workflow_steps.append({
                                            "agent_id": agent_id,
                                            "refined_query": refined_query
                                        })
                        all_workflows[workflow_id] = workflow_steps
            
            return all_workflows
            
        except Exception as e:
            logger.debug(f"⚠️ Error parsing workflow: {e}", style="yellow")
            return {}
    

    def remove_reasoning_columns(self, msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        keys = ["agent_id","role","content","tool_calls"]
        new_msgs = []
        for msg in msgs:
            logger.debug( f"Removing reasoning columns from message: {msg}")
            new_msg = {}
            for key in keys:
                if key in msg:
                    new_msg[key] = msg[key]
            new_msgs.append(new_msg)
        return new_msgs

    def parse_tool_content(self, content: str, agent_id: str, step_id: int) -> Dict[str, Dict[str, Any]]:
        """
        Parse tool content and convert to OpenAI format with numbered keys.
        
        Args:
            content: Tool content string
            agent_id: Agent ID
            step_id: Step ID
            
        Returns:
            Dictionary with numbered keys containing message dictionaries in OpenAI format
        """
        messages = {}
        
        try:
            yaml_content = yaml.safe_load(content.strip())
            # Create tool call message
            tool_result = yaml_content.get("tool_call_result", "")
            arguments = yaml_content.get("arguments") if "arguments" in yaml_content else yaml_content.get("parameters", {})
            tool_name = yaml_content.get("tool", "tool_name")
            tool_calls = yaml_content.get("tool_call", [])
            tool_call_id = generate_custom_hash()
            if not tool_calls:
                tool_call_message = {
                    "role": "assistant",
                    "agent_id": agent_id,
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": tool_call_id,
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(arguments, ensure_ascii=False) if arguments else "{}"
                            }
                        }
                    ]
                }
            else:
                tool_name = tool_calls[0].get("tool", "tool_name")
                tool_call_message = {
                    "role": "assistant",
                    "agent_id": agent_id,
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": tool_call_id,
                            "function": {
                                "name": single_toolcall.get('tool', tool_name),
                                "arguments": json.dumps(single_toolcall.get('arguments',{}), ensure_ascii=False)
                                }
                            }  for i, single_toolcall in enumerate(tool_calls)
                        ]
                    }
            # Create tool result message
            tool_result_message = {
                "role": "tool",
                "agent_id": agent_id,
                "tool_call_id": tool_call_id,
                "content": tool_result,
                "name": tool_name
            }
            
            # Add messages with numbered keys in format {step_id}-1, {step_id}-2
            messages[f"{step_id}-1"] = tool_call_message
            messages[f"{step_id}-2"] = tool_result_message
            
        except Exception as e:
            # Fallback: create simple assistant message
            messages[f"{step_id}-1"] = {
                "role": "assistant",
                "agent_id": agent_id,
                "content": content
            }
        
        return messages
    
    def get_non_agent_messages_from_label(self, label: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get all messages without agent_id from label dictionary.
        
        Args:
            label: Dictionary of messages
            
        Returns:
            List of messages without agent_id
        """
        non_agent_messages = []
        
        # Sort by keys to maintain order
        sorted_keys = sorted(label.keys(), key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1]) if '-' in x else 0))
        
        for key in sorted_keys:
            msg = label[key]
            if "agent_id" not in msg:
                non_agent_messages.append(msg)
        
        return non_agent_messages

    def build_history_up_to_step(self, history_label: Dict[str, Dict[str, Any]], target_step: int) -> List[Dict[str, Any]]:
        """
        Build conversation history up to (but not including) the target step.
        
        Args:
            history_label: The label dictionary containing all messages
            target_step: The step number to stop before
            
        Returns:
            List of messages in OpenAI format for agent execution
        """
        conversation_history = []
        
        # Get all keys before target step and sort them
        relevant_keys = [k for k in history_label.keys() 
                        if int(str(k).split('-')[0]) < target_step]
        sorted_keys = sorted(relevant_keys, 
                           key=lambda x: (int(str(x).split('-')[0]), 
                                        int(str(x).split('-')[1]) if '-' in str(x) else 0))
        
        for key in sorted_keys:
            msg = history_label[key].copy()
            
            # Remove agent_id for clean history
            if "agent_id" in msg:
                msg.pop("agent_id", None)
                
                # Convert tool messages to assistant messages for compatibility
                if msg.get("role") == "tool":
                    tmp_message = {
                        "role": "assistant", 
                        "content": json.dumps(msg["content"], ensure_ascii=False)
                    }
                    conversation_history.append(tmp_message)
            elif msg.get("role") in ["user", "assistant"]:
                # if "workflow_" in msg.get("content", ""):
                #     # If content contains workflow, convert to system message
                #     json_msg = yaml.safe_load(msg["content"])
                #     msg["content"] = "Previous generated workflow : \n\n" + json.dumps(json_msg, ensure_ascii=False)
                conversation_history.append(msg)
        
        return conversation_history

    def find_workflow_step_for_agent(self, agent_id: str, all_workflows: Dict[str, List[Dict[str, str]]], 
                                   used_workflow_indices: Dict[str, set]) -> Optional[str]:
        """
        Find the next unused workflow step for the given agent.
        
        Args:
            agent_id: The agent ID to find a step for
            all_workflows: Dictionary of all parsed workflows
            used_workflow_indices: Tracking of used workflow step indices
            
        Returns:
            The refined_query for the agent, or None if not found
        """
        for workflow_id, workflow_steps in all_workflows.items():
            used_indices = used_workflow_indices.get(workflow_id, set())
            
            for idx, workflow_step in enumerate(workflow_steps):
                if (workflow_step["agent_id"] == agent_id and 
                    idx not in used_indices):
                    # Mark this step as used
                    used_indices.add(idx)
                    used_workflow_indices[workflow_id] = used_indices
                    return workflow_step["refined_query"]
        
        return None

    async def execute_agents_batch(
        self, 
        batch_tasks: List[Tuple[EvaluationStep, Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, str]]], Dict[str, set], str]], 
        batch_size: int = 5,
        max_retries: int = 1,
        retry_delay: float = 1.0
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Execute multiple agents in batch for better performance with configurable batch size.
        
        Args:
            batch_tasks: List of tuples containing (step, history_label, all_workflows, used_workflow_indices, system_info)
            batch_size: Maximum number of concurrent executions per batch
            max_retries: Maximum number of retry attempts for failed tasks
            retry_delay: Delay in seconds between retry attempts
            
        Returns:
            List of history entries (same order as input)
        """
        if not batch_tasks:
            return []
        
        # 입력 검증
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        
        logger.debug(f"🚀 Executing {len(batch_tasks)} agents in batches of {batch_size}...", style="bold yellow")
        logger.debug(f"  🔧 Retry configuration: max_retries={self.max_retries}")
        
        all_results = []
        
        # Semaphore를 사용한 동시성 제어 (옵션)
        semaphore = asyncio.Semaphore(batch_size * 2)  # 전체 동시 실행 제한
        
        async def execute_with_semaphore(task_data, task_index):
            """Semaphore를 사용한 task 실행 wrapper"""
            async with semaphore:
                try:
                    result = await self.execute_agent_and_record(*task_data)
                    return result, task_index
                except Exception as e:
                    step = task_data[0]
                    logger.debug(f"    💥 Exception in execute_with_semaphore for {step.agent}: {type(e).__name__}: {e}", style="red")
                    return e, task_index
        
        # Process tasks in chunks of batch_size
        for i in range(0, len(batch_tasks), batch_size):
            batch_chunk = batch_tasks[i:i + batch_size]
            chunk_number = (i // batch_size) + 1
            total_chunks = (len(batch_tasks) + batch_size - 1) // batch_size
            
            logger.debug(f"  📦 Processing batch {chunk_number}/{total_chunks} ({len(batch_chunk)} tasks)...", style="yellow")
            
            # Create async tasks for this chunk with indices
            tasks = []
            for j, task_data in enumerate(batch_chunk):
                task = asyncio.create_task(execute_with_semaphore(task_data, j))
                tasks.append(task)
            
            # Execute chunk tasks concurrently
            chunk_results = [None] * len(batch_chunk)  # 결과를 순서대로 저장
            failed_indices = []
            
            try:
                # gather with return_exceptions=True를 사용하여 예외도 결과로 받음
                raw_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results - 각 raw_result는 (result, idx) 튜플이거나 Exception임
                for i, raw_result in enumerate(raw_results):
                    if isinstance(raw_result, Exception):
                        # Task 자체가 예외를 발생시킨 경우
                        step = batch_chunk[i][0]
                        logger.debug(f"    💥 Task execution exception for {step.agent}: {raw_result}", style="red")
                        failed_indices.append(i)
                    elif isinstance(raw_result, tuple) and len(raw_result) == 2:
                        result, idx = raw_result
                        if isinstance(result, Exception):
                            step = batch_chunk[idx][0]
                            logger.debug(f"    ⚠️ Task failed for {step.agent}: {type(result).__name__}: {result}", style="yellow")
                            failed_indices.append(idx)
                        else:
                            chunk_results[idx] = result
                            step = batch_chunk[idx][0]
                            logger.debug(f"    ✅ Task succeeded for {step.agent}", style="green")
                    else:
                        # 예상치 못한 결과 형태 - 인덱스 기반으로 실패 처리
                        step = batch_chunk[i][0]
                        logger.debug(f"    ❓ Unexpected result format for {step.agent}: {raw_result}", style="yellow")
                        failed_indices.append(i)
                        
            except Exception as e:
                # gather 자체가 실패한 경우 (return_exceptions=False일 때)
                logger.debug(f"  ❌ Batch {chunk_number} execution failed critically: {e}", style="red")
                # 모든 task를 실패로 처리
                failed_indices = list(range(len(batch_chunk)))
            
            # Retry failed tasks if any
            if failed_indices and max_retries > 0:
                failed_agents = [batch_chunk[idx][0].agent for idx in failed_indices]
                logger.debug(f"  🔄 Retrying {len(failed_indices)} failed tasks (max_retries={max_retries}) for agents: {failed_agents}", style="yellow")
                
                for retry_attempt in range(max_retries):
                    if not failed_indices:
                        break
                        
                    logger.debug(f"    🔁 Retry attempt {retry_attempt + 1}/{max_retries} for {len(failed_indices)} tasks", style="cyan")
                    await asyncio.sleep(retry_delay * (retry_attempt + 1))  # Exponential backoff
                    
                    retry_tasks = []
                    for idx in failed_indices:
                        task_data = batch_chunk[idx]
                        retry_tasks.append((task_data, idx))
                    
                    # Retry with smaller batch size for better success rate
                    retry_batch_size = max(1, batch_size // 2)
                    still_failed = []
                    
                    for retry_idx in range(0, len(retry_tasks), retry_batch_size):
                        retry_chunk = retry_tasks[retry_idx:retry_idx + retry_batch_size]
                        
                        retry_coroutines = []
                        for task_data, original_idx in retry_chunk:
                            retry_coroutines.append(self.execute_agent_and_record(*task_data))
                        
                        try:
                            retry_results = await asyncio.gather(*retry_coroutines, return_exceptions=True)
                            
                            for (_, original_idx), result in zip(retry_chunk, retry_results):
                                if isinstance(result, Exception):
                                    still_failed.append(original_idx)
                                    logger.debug(f"      ❌ Retry failed for index {original_idx}: {result}", style="red")
                                else:
                                    chunk_results[original_idx] = result
                                    logger.debug(f"      ✅ Retry succeeded for index {original_idx}", style="green")
                        except Exception as retry_error:
                            logger.debug(f"      ❌ Retry batch failed: {retry_error}", style="red")
                            still_failed.extend([idx for _, idx in retry_chunk])
                    
                    failed_indices = still_failed
            elif failed_indices:
                failed_agents = [batch_chunk[idx][0].agent for idx in failed_indices]
                logger.debug(f"  ⚠️ {len(failed_indices)} tasks failed but max_retries=0, skipping retry for agents: {failed_agents}", style="yellow")
            
            # Log final status for this chunk
            success_count = sum(1 for r in chunk_results if r is not None)
            logger.debug(f"  ✅ Batch {chunk_number} completed: {success_count}/{len(batch_chunk)} successful", style="green")
            
            all_results.extend(chunk_results)
            
            # Small delay between chunks to prevent overwhelming the API
            if i + batch_size < len(batch_tasks):
                await asyncio.sleep(0.5)
        
        # Final statistics
        total_success = sum(1 for r in all_results if r is not None)
        total_failed = len(all_results) - total_success
        
        logger.debug(f"🎯 Total batch execution completed: {total_success}/{len(batch_tasks)} successful", style="bold green")
        
        if total_failed > 0:
            logger.debug(f"⚠️  {total_failed} tasks failed after all retry attempts", style="bold yellow")
        
        return all_results
        
    async def execute_agent_and_record(self, step: EvaluationStep, history_label: Dict[str, Dict[str, Any]], 
                                        all_workflows: Dict[str, List[Dict[str, str]]], 
                                        used_workflow_indices: Dict[str, set], system_info: str) -> Optional[Dict[str, Any]]:
        """
        Execute agent and record the result in sub_agent_history.
        
        Args:
            step: Current evaluation step
            history_label: Current history label
            all_workflows: All parsed workflows
            used_workflow_indices: Tracking of used workflow indices
            system_info: System information
            
        Returns:
            Dictionary to add to sub_agent_history, or None if execution failed
        """
        agent_id = step.agent
        base_step_id = int(str(step.step_id).split('-')[0]) if '-' in str(step.step_id) else step.step_id
        
        # Find matching workflow step
        user_query_from_workflow = self.find_workflow_step_for_agent(
            agent_id, all_workflows, used_workflow_indices
        )
        
        # Build history up to step before base_step_id
        agent_history = self.build_history_up_to_step(history_label, base_step_id)
        
        # # Add user query from workflow if found
        if user_query_from_workflow:
            user_query = {"role": "user", "content": user_query_from_workflow}
            agent_history.append(user_query)
            logger.debug(f"  📝 Added workflow query: {user_query_from_workflow}", style="blue")
        
        # Call agent execution
        try:
            agent = self.orchestration_engine.agents.get(agent_id)
            if agent:
                await agent.initialize()
                if agent.model is None:
                    raise RuntimeError(f"Model for agent {agent_id} is not initialized")
                

                logger.debug(f"Agent tools: {agent.tools}")
                response = await agent._complete_general(agent_history, system_info=system_info)
                
                # Create history entry for sub_agent_history
                history_entry = {"step_id": f"{step.step_id}"}
                if isinstance(response, list) and response:
                    # Handle list of responses
                    for item in response:
                        if isinstance(item, dict):
                            item["agent_id"] = agent_id
                    cleaned_response = self.remove_reasoning_columns(response)[0] if response else {}
                    history_entry.update(cleaned_response)
                elif isinstance(response, dict):
                    response["agent_id"] = agent_id
                    cleaned_response = self.remove_reasoning_columns([response])[0]
                    history_entry.update(cleaned_response)
                else:
                    # Handle string or other response types
                    history_entry.update({
                        "role": "assistant", 
                        "content": str(response),
                        "agent_id": agent_id
                    })
                
                logger.debug(f"  🤖 Agent execution completed for {agent_id}, added to sub_agent_history", style="green")
                return history_entry
            else:
                logger.debug(f"  ❌ Agent {agent_id} not found", style="red")
                return None
                
        except Exception as e:
            logger.debug(f"  ❌ Agent execution failed for {agent_id}: {type(e).__name__}: {e}", style="red")
            logger.debug(f"  📋 Full traceback:\n{traceback.format_exc()}", style="dim")
            # Check if it's an HTTP error
            if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                logger.debug(f"  🌐 HTTP Error: {e.response.status_code}", style="red")
            return None

    async def build_history_from_steps(self, evaluation_steps: List[EvaluationStep]) -> Tuple[Dict[str, Any], str, Dict[str, List[Dict[str, str]]]]:
        """
        Build history structure from evaluation steps and execute agents as needed.
        
        Args:
            evaluation_steps: List of EvaluationStep objects
            
        Returns:
            Tuple containing (history dict, system_info, all_workflows)
        """
        # Initialize history structure
        history = {
            "main_agent_history": [],
            "sub_agent_history": [],
            "label": {}  # Changed to dictionary
        }
        
        system_info = ""
        all_workflows = {}  # workflow_id -> list of steps
        
        # Track used workflow step indices for each workflow
        used_workflow_indices = {}  # workflow_id -> set of used indices
        
        # Collect batch execution tasks
        batch_execution_tasks = []  # List of (step, agent_history, user_query) tuples
        
        logger.debug("🏗️ Building history structure from evaluation steps...", style="bold cyan")
        
        for i, step in enumerate(evaluation_steps):
            logger.debug(f"Processing step {step.step_id}: {step.agent} ({step.agent_type}) - workflow_id: {step.workflow_id}", style="dim")
            
            content = step.data.get("content", "")
            
            # 1. workflow_id='system' - store system_info
            if step.workflow_id == "system":
                system_info = content
                # Add system message to label with step_id as key
                history["label"][step.step_id] = {"role": "system", "content": content}
                logger.debug("  📋 Stored system info and added to label", style="green")
                                
            # 2. role=assistant and content starts with workflow_ - call workflow generation
            elif content.startswith("workflow_"):
                # Add workflow message to label first
                workflow_message = {"role": "assistant", "content": content}
                history["label"][step.step_id] = workflow_message
                
                # Get history up to current step for workflow generation
                current_history = self.build_history_up_to_step(history["label"], step.step_id)
                
                # Call workflow generation with retry logic
                workflow_response = None
                for attempt in range(self.max_retries + 1):
                    try:
                        logger.debug(f"  🔄 Workflow generation attempt {attempt + 1}/{self.max_retries + 1}")
                        response = await self.orchestration_agent.generate_workflow(
                            msgs=current_history,
                            system_info=system_info
                        )
                        
                        # Add to main_agent_history with step_id
                        history_entry = {"step_id": step.step_id}
                        if isinstance(response, dict):
                            history_entry.update(self.remove_reasoning_columns([response])[0])
                        else:
                            history_entry.update({"role": "assistant", "content": str(response)})
                        history["main_agent_history"].append(history_entry)
                        
                        logger.debug("  🔄 Workflow generation completed", style="green")
                        workflow_response = response
                        break
                        
                    except Exception as e:
                        error_msg = str(e)
                        logger.debug(f"  ❌ Workflow generation attempt {attempt + 1} failed: {error_msg}")
                        
                        # Check if this is a retryable error
                        is_retryable = (
                            "500" in error_msg or
                            "timeout" in error_msg.lower() or
                            "connection" in error_msg.lower() or
                            "network" in error_msg.lower()
                        )
                        
                        if not is_retryable or attempt == self.max_retries:
                            logger.debug(f"  💥 Workflow generation failed after {attempt + 1} attempts: {error_msg}")
                            if is_retryable:
                                logger.debug(f"  🔄 Max retries ({self.max_retries}) exceeded for workflow generation")
                            else:
                                logger.debug(f"  🚫 Non-retryable error for workflow generation: {error_msg}")
                            traceback.print_exc()
                            break
                        
                        # Exponential backoff before retry
                        delay = min(2 ** attempt, 60)
                        logger.debug(f"  ⏳ Retrying workflow generation in {delay} seconds...")
                        await asyncio.sleep(delay)
                
                # Parse all workflows from content
                parsed_workflows = self.parse_workflow(content)
                all_workflows.update(parsed_workflows)
                
                # Initialize used indices tracker for each new workflow
                for workflow_id in parsed_workflows.keys():
                    if workflow_id not in used_workflow_indices:
                        used_workflow_indices[workflow_id] = set()
                
                logger.debug(f"  📝 Parsed {len(parsed_workflows)} workflows with total {sum(len(steps) for steps in parsed_workflows.values())} total steps", style="green")
            
            # 3. Non-main, non-user agents - collect for batch processing
            elif step.agent_type not in ["main", "user"]:
                agent_id = step.agent
                
                # Add content to label first
                message = {
                    "role": "assistant",
                    "content": content,
                    "agent_id": agent_id
                }
                history["label"][step.step_id] = message
                
                # Check if this is a tool call
                if "tool" in content:
                    # Parse tool content and update label
                    tool_messages = self.parse_tool_content(content, agent_id, step.step_id)
                    history["label"].update(tool_messages)
                    logger.debug(f"  🔧 Parsed tool content, added {len(tool_messages)} tool messages", style="yellow")
                    
                    # Collect for batch execution if this is a tool call message (ends with -1 or no dash)
                    if (str(step.step_id).endswith('-1') or '-' not in str(step.step_id)):
                        batch_execution_tasks.append((step, history["label"], all_workflows, used_workflow_indices, system_info))
                else:
                    # Non-tool response - collect for batch execution
                    logger.debug("  💬 Collecting non-tool agent response for batch execution", style="cyan")
                    batch_execution_tasks.append((step, history["label"], all_workflows, used_workflow_indices, system_info))
        
            # 4. agent='main' - add assistant message to label
            elif step.agent == "main":
                message = {"role": "assistant", "content": content}
                history["label"][step.step_id] = message
                logger.debug("  🤖 Added main agent message", style="green")
            
            # 5. agent='user' - add user message to label
            elif step.agent == "user":
                message = {"role": "user", "content": content}
                history["label"][step.step_id] = message
                logger.debug("  👤 Added user message", style="blue")
        
        # Execute all collected agent tasks in batch
        if batch_execution_tasks:
            logger.debug(f"🚀 Executing {len(batch_execution_tasks)} agents in batches of {self.batch_size}...", style="bold yellow")
            batch_results = await self.execute_agents_batch(batch_execution_tasks, self.batch_size, self.max_retries)
            
            # Add successful results to sub_agent_history
            for result in batch_results:
                if result is not None:
                    history["sub_agent_history"].append(result)
        
        total_workflow_steps = sum(len(steps) for steps in all_workflows.values())
        logger.debug(f"✅ History structure built: {len(history['label'])} messages, main_agent: {len(history['main_agent_history'])}, sub_agent: {len(history['sub_agent_history'])} across {len(all_workflows)} workflows with {total_workflow_steps} total steps", style="bold green")
        return history, system_info, all_workflows
    
    async def process_evaluation_steps(self, evaluation_steps: List[EvaluationStep]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Process evaluation steps and generate history according to the specified workflow.
        
        Args:
            evaluation_steps: List of EvaluationStep objects
            
        Returns:
            Tuple containing (history dictionary, usage information)
        """
        logger.debug("🎯 Starting evaluation step processing...", style="bold cyan")
        
        # Build history structure and execute agents as needed
        history, system_info, workflow_steps = await self.build_history_from_steps(evaluation_steps)
        
        # Get usage information
        usage = self.orchestration_engine.get_token_usage_summary()
        
        logger.debug("🎉 Evaluation step processing completed", style="bold green")
        return history, usage

    def display_history(self, history: Dict[str, Any], usage: Dict[str, Any]):
        """Display all history components"""
        logger.debug("\n" + "="*80, style="bold cyan")
        logger.debug("📊 FINAL HISTORY DISPLAY", style="bold cyan")
        logger.debug("TOKEN USAGE {}", usage, style="bold cyan")
        logger.debug("="*80, style="bold cyan")
        
        # Display main_agent_history
        logger.debug(f"\n🤖 MAIN AGENT HISTORY ({len(history['main_agent_history'])} items)", style="bold green")
        logger.debug("-" * 60, style="dim")
        for i, item in enumerate(history["main_agent_history"], 1):
            logger.debug(f"{i:2d}. {{}}", item, style="green")

        # Display sub_agent_history
        logger.debug(f"\n🔧 SUB AGENT HISTORY ({len(history['sub_agent_history'])} items)", style="bold magenta")
        logger.debug("-" * 60, style="dim")
        for i, item in enumerate(history["sub_agent_history"], 1):
            agent_info = f"[{item.get('agent', item.get('agent_id', 'unknown'))}]" if 'agent' in item or 'agent_id' in item else ""
            logger.debug(f"{i:2d}. {{}}", item, style="magenta")

        logger.debug(f"\n🏷️  LABEL HISTORY ({len(history['label'])} items)", style="bold blue")
        logger.debug("-" * 60, style="dim")
        
        # Sort dictionary keys for display
        sorted_keys = sorted(history['label'].keys(), key=lambda x: (int(str(x).split('-')[0]), int(str(x).split('-')[1]) if '-' in str(x) else 0))
        
        for i, key in enumerate(sorted_keys, 1):
            item = history['label'][key]
            role = item.get("role", "unknown")
            agent_info = f"[{item.get('agent', item.get('agent_id', ''))}]" if 'agent' in item or 'agent_id' in item else ""
            
            # Role styling
            if role == "system":
                role_icon = "⚙️"
                role_color = "cyan"
            elif role == "user":
                role_icon = "👤"
                role_color = "blue"
            elif role == "assistant":
                role_icon = "🤖"
                role_color = "green"
            elif role == "tool":
                role_icon = "🔧"
                role_color = "yellow"
            else:
                role_icon = "❓"
                role_color = "dim"
            
            content = item.get("content", "")
            if len(content) > 100:
                content = content[:100] + "..."

            logger.debug(f"{key}. {{}}", role_icon, style=f"bold {role_color}")
            logger.debug(f"     {{}}", content, style=role_color)

            # Show additional fields
            for field_key, value in item.items():
                if field_key not in ["role", "content", "agent", "agent_id"]:
                    logger.debug("     {}: {}", field_key, value, style="dim")

        logger.debug("="*80, style="bold cyan")

# Convenience function for easy usage
async def generate_simple_history(
    evaluation_steps: List[EvaluationStep], 
    config_file: Path, 
    agent_cards_dir: Path,
    model_type: str = "openai",
    batch_size: int = 5,
    max_retries: int = 1
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Convenience function to generate history from evaluation steps.
    
    Args:
        evaluation_steps: List of EvaluationStep objects
        config_file: Path to the configuration YAML file
        agent_cards_dir: Path to agent cards directory
        model_type: Type of model to use
        batch_size: Maximum number of concurrent agent executions per batch
        max_retries: Maximum number of retry attempts for failed agent executions
        
    Returns:
        Tuple containing (history dictionary, usage information)
    """
    generator = SimpleHistoryGenerator(config_file, agent_cards_dir, model_type, batch_size, max_retries)
    await generator.initialize()
    history, usage = await generator.process_evaluation_steps(evaluation_steps)
    generator.display_history(history, usage)
    return history, usage

# Example usage
if __name__ == "__main__":
    import sys 
    async def main():
        # Sample evaluation steps (as provided in the request)
        evaluation_steps = [
            EvaluationStep(workflow_id='system', step_id=1, agent='main', agent_type='main', 
                         data={'content': '현재 날짜: 2025-07-09\n현재 시간: 15:00\n장소: 경기도 성남시 삼평동'}),
            EvaluationStep(workflow_id='user', step_id=2, agent='user', agent_type='user', 
                         data={'content': '내일 고객사 미팅 일정 후에 바로 볼만한 영화 뭐가 있을까? 예매하고 싶어.'}),
            EvaluationStep(workflow_id=None, step_id=3, agent='main', agent_type='main', 
                         data={'content': 'workflow_1:\n  type: independent\n  steps:\n    - name: calendar_agent\n      refined_query: "내일 고객사 미팅 일정 시간 조회"\n\n    - name: entertainment_agent\n      refined_query: "고객사 미팅 이후 근처에서 볼 수 있는 상영 영화 추천"\n\n    - name: entertainment_agent\n      refined_query: "고객사 미팅 일정 이후 볼만한 영화 예매"'}),
            EvaluationStep(workflow_id='workflow_1', step_id=4, agent='calendar_agent', agent_type='calendar_agent', 
                         data={'content': 'tool: getSchedule\nargument:\n  refinedQuery: "내일 고객사 미팅 일정"\n  startDate: "2025-07-10"\n  endDate: "2025-07-10"\n  keyword: "고객사 미팅"\ntool_call_result:\n  data:\n    scheduleID: "evt_30711"\n    title: "고객사 미팅"\n    startDateTime: "2025-07-10T14:00"\n    endDateTime: "2025-07-10T17:30"\n    location: "강남역 5번 출구 근처 위워크"'}),
            EvaluationStep(workflow_id=None, step_id=5, agent='entertainment_agent', agent_type='agent', 
                         data={'content': 'tool: getMovieInfo\nargument:\n  refinedQuery: "강남역 근처에서 7월 10일 오후 5시 이후 관람 가능한 인기 영화 추천"\n  isReleased: true\n  theaterLocation: "강남역"\n  releaseDate: "2025-07-10"\n  sortBy: "popularity"\ntool_call_result:\n  data:\n    movies:\n      - title: "파묘"\n        genre: "스릴러"\n        duration: "120분"\n        rating: 4.5\n        showtimes:\n          - time: "18:20"\n            theater: "롯데시네마 강남"\n          - time: "20:45"\n            theater: "CGV 강남"\n\n      - title: "인사이드 아웃 2"\n        genre: "애니메이션"\n        duration: "105분"\n        rating: 4.7\n        showtimes:\n          - time: "18:00"\n            theater: "CGV 강남"\n          - time: "19:40"\n            theater: "메가박스 코엑스"\n\n      - title: "범죄도시 4"\n        genre: "액션"\n        duration: "110분"\n        rating: 4.2\n        showtimes:\n          - time: "17:50"\n            theater: "롯데시네마 강남"\n          - time: "20:10"\n            theater: "메가박스 코엑스"\n\n      - title: "쿵푸팬더 5"\n        genre: "애니메이션/코미디"\n        duration: "95분"\n        rating: 4.4\n        showtimes:\n          - time: "18:30"\n            theater: "CGV 강남"\n          - time: "20:00"\n            theater: "메가박스 강남"\n\n      - title: "듄: 파트2"\n        genre: "SF"\n        duration: "150분"\n        rating: 4.6\n        showtimes:\n          - time: "19:00"\n            theater: "메가박스 코엑스"\n          - time: "21:20"\n            theater: "CGV 강남"'}),
            EvaluationStep(workflow_id=None, step_id=6, agent='main', agent_type='main', 
                         data={'content': '아래는 추천 영화와 상영 스케줄입니다. 어떤 영화의 어느 상영관, 몇 시 영화를 예매할까요?\n\n🎬 **파묘** (스릴러, 120분, ★4.5)\n- 롯데시네마 강남: 오후 6시 20분\n- CGV 강남: 오후 8시 45분\n\n🎬 **인사이드 아웃 2** (애니메이션, 105분, ★4.7)\n- CGV 강남: 오후 6시\n- 메가박스 코엑스: 오후 7시 40분\n\n🎬 **범죄도시 4** (액션, 110분, ★4.2)\n- 롯데시네마 강남: 오후 5시 50분\n- 메가박스 코엑스: 오후 8시 10분\n\n🎬 **쿵푸팬더 5** (애니메이션/코미디, 95분, ★4.4)\n- CGV 강남: 오후 6시 30분\n- 메가박스 강남: 오후 8시\n\n🎬 **듄: 파트2** (SF, 150분, ★4.6)\n- 메가박스 코엑스: 오후 7시\n- CGV 강남: 오후 9시 20분'}),
            EvaluationStep(workflow_id=None, step_id=7, agent='user', agent_type='user', 
                         data={'content': '인사이드 아웃 2 CGV 강남 6시로 할게.'}),
            EvaluationStep(workflow_id=None, step_id=8, agent='entertainment_agent', agent_type='agent', 
                         data={'content': '예매자 이름, 연락처, 인원 수 필요'}),
            EvaluationStep(workflow_id=None, step_id=9, agent='main', agent_type='main', 
                         data={'content': '예약을 위해 고객님의 이름, 연락처, 관람 인원을 알려주세요.'}),
            EvaluationStep(workflow_id=None, step_id=10, agent='user', agent_type='user', 
                         data={'content': '이름은 박지나, 010-1222-2222야. 혼자 볼 거야.'}),
            EvaluationStep(workflow_id=None, step_id=11, agent='entertainment_agent', agent_type='agent', 
                         data={'content': 'tool: bookMovieTicket\nargument:\n  refinedQuery: "7월 10일 18:00 CGV 강남 인사이드 아웃 2 예매"\n  movieTitle: "인사이드 아웃 2"\n  theaterName: "CGV 강남"\n  showDate: "2025-07-10"\n  showTime: "18:00"\n  audienceCount: 1\n  audienceTypes: "adult"\n  userInfo:\n    name: "박지나"\n    contactNumber: "010-1222-2222"\ntool_call_result:\n  data:\n    bookingId: "MOV20250710_142"\n    movieTitle: "인사이드 아웃 2"\n    theater: "CGV 강남"\n    showTime: "2025-07-10T18:00"\n    seat: "E7"\n    status: "confirmed"'}),
            EvaluationStep(workflow_id=None, step_id=12, agent='main', agent_type='main', 
                         data={'content': '예매 완료! 🎟️\n\n- 영화: 인사이드 아웃 2\n- 장소: CGV 강남\n- 시간: 7월 10일 오후 6시\n- 좌석: E7\n- 예약번호: MOV20250710_142\n\n즐거운 관람되세요. 😊'})
        ]
        
        # Default paths
        config_file = Path("config/base_config/multiagent_config.yaml")
        agent_cards_dir = Path("config/multiagent_cards")
        model_type = "gemini"
        try:
            history, usage = await generate_simple_history(evaluation_steps, config_file, agent_cards_dir, model_type)
            for k, v in history.items():
                logger.debug("     {}: {}", k, v, style="dim")

        except Exception as e:
            traceback.print_exc()
            logger.debug(f"💥 History generation failed: {e}", style="bold red")
            sys.exit(1)
    
    asyncio.run(main())