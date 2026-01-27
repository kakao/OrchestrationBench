"""
Centralized token usage tracking for the orchestration system.

This module provides a centralized way to track token usage across all model calls
including those made directly by OrchestrationEngine and those made within WorkflowHandler
through agent executions.
"""
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import threading
from loguru import logger

@dataclass
class TokenUsage:
    """Single token usage record."""
    timestamp: datetime
    model_name: str
    agent_id: Optional[str]
    workflow_id: Optional[str]
    operation: str  # e.g., "workflow_generation", "agent_execution", "classification"
    input_tokens: int
    output_tokens: int
    cost: float = 0.0
    
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

@dataclass
class TokenSummary:
    """Summary of token usage statistics."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    total_calls: int = 0
    
    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

class TokenTracker:
    """
    Centralized token usage tracker for the orchestration system.
    
    This class collects token usage from all models across the system including:
    - OrchestrationEngine direct calls (workflow generation, intent classification)
    - WorkflowHandler agent executions
    - Any other model calls throughout the system
    """
    
    def __init__(self):
        self.usage_records: List[TokenUsage] = []
        self._lock = threading.Lock()
    
    def track_usage(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        cost: float = 0.0,
        agent_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        operation: str = "unknown"
    ) -> None:
        """
        Track token usage from any model call.
        
        Args:
            model_name: Name of the model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost: Cost of the operation in USD
            agent_id: ID of the agent making the call (if applicable)
            workflow_id: ID of the workflow (if applicable)
            operation: Type of operation (e.g., "workflow_generation", "agent_execution")
        """
        with self._lock:
            usage = TokenUsage(
                timestamp=datetime.now(),
                model_name=model_name,
                agent_id=agent_id,
                workflow_id=workflow_id,
                operation=operation,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost
            )
            self.usage_records.append(usage)
            
            logger.debug(f"🔢 TokenTracker: {model_name} - {input_tokens}/{output_tokens} tokens, ${cost:.4f} ({operation})")
    
    def get_total_summary(self) -> TokenSummary:
        """Get total token usage summary across all operations."""
        with self._lock:
            summary = TokenSummary()
            for record in self.usage_records:
                summary.total_input_tokens += record.input_tokens
                summary.total_output_tokens += record.output_tokens
                summary.total_cost += record.cost
                summary.total_calls += 1
            return summary
    
    def get_summary_by_agent(self) -> Dict[str, TokenSummary]:
        """Get token usage summary grouped by agent."""
        with self._lock:
            summaries: Dict[str, TokenSummary] = {}
            for record in self.usage_records:
                agent_key = record.agent_id or "orchestrator"
                if agent_key not in summaries:
                    summaries[agent_key] = TokenSummary()
                
                summary = summaries[agent_key]
                summary.total_input_tokens += record.input_tokens
                summary.total_output_tokens += record.output_tokens
                summary.total_cost += record.cost
                summary.total_calls += 1
            
            return summaries
    
    def get_summary_by_model(self) -> Dict[str, TokenSummary]:
        """Get token usage summary grouped by model."""
        with self._lock:
            summaries: Dict[str, TokenSummary] = {}
            for record in self.usage_records:
                if record.model_name not in summaries:
                    summaries[record.model_name] = TokenSummary()
                
                summary = summaries[record.model_name]
                summary.total_input_tokens += record.input_tokens
                summary.total_output_tokens += record.output_tokens
                summary.total_cost += record.cost
                summary.total_calls += 1
            
            return summaries
    
    def get_summary_by_operation(self) -> Dict[str, TokenSummary]:
        """Get token usage summary grouped by operation type."""
        with self._lock:
            summaries: Dict[str, TokenSummary] = {}
            for record in self.usage_records:
                if record.operation not in summaries:
                    summaries[record.operation] = TokenSummary()
                
                summary = summaries[record.operation]
                summary.total_input_tokens += record.input_tokens
                summary.total_output_tokens += record.output_tokens
                summary.total_cost += record.cost
                summary.total_calls += 1
            
            return summaries
    
    def get_detailed_usage(
        self, 
        agent_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        operation: Optional[str] = None
    ) -> List[TokenUsage]:
        """
        Get detailed usage records with optional filtering.
        
        Args:
            agent_id: Filter by agent ID
            workflow_id: Filter by workflow ID
            operation: Filter by operation type
            
        Returns:
            List of matching TokenUsage records
        """
        with self._lock:
            filtered_records = []
            for record in self.usage_records:
                if agent_id and record.agent_id != agent_id:
                    continue
                if workflow_id and record.workflow_id != workflow_id:
                    continue
                if operation and record.operation != operation:
                    continue
                filtered_records.append(record)
            
            return filtered_records
    
    def reset(self) -> None:
        """Reset all tracking data."""
        with self._lock:
            self.usage_records.clear()
            logger.debug("🔢 TokenTracker: Reset all tracking data")
    
    def create_callback(
        self, 
        agent_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        operation: str = "model_call"
    ) -> Callable[[str, int, int, float], None]:
        """
        Create a callback function for models to report token usage.
        
        Args:
            agent_id: Agent ID to associate with calls
            workflow_id: Workflow ID to associate with calls  
            operation: Operation type to associate with calls
            
        Returns:
            Callback function that models can call to report usage
        """
        def callback(model_name: str, input_tokens: int, output_tokens: int, cost: float = 0.0):
            self.track_usage(
                model_name=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                agent_id=agent_id,
                workflow_id=workflow_id,
                operation=operation
            )
        
        return callback
    
    def print_summary(self) -> None:
        """Print a formatted summary of token usage."""
        total = self.get_total_summary()
        by_agent = self.get_summary_by_agent()
        by_model = self.get_summary_by_model()
        by_operation = self.get_summary_by_operation()
        
        print("\n" + "="*60)
        print("🔢 TOKEN USAGE SUMMARY")
        print("="*60)
        
        print(f"\n📊 TOTAL USAGE:")
        print(f"   Total Calls: {total.total_calls}")
        print(f"   Input Tokens: {total.total_input_tokens:,}")
        print(f"   Output Tokens: {total.total_output_tokens:,}")
        print(f"   Total Tokens: {total.total_tokens:,}")
        print(f"   Total Cost: ${total.total_cost:.4f}")
        
        print(f"\n🤖 BY AGENT:")
        for agent_id, summary in sorted(by_agent.items()):
            print(f"   {agent_id}: {summary.total_tokens:,} tokens, ${summary.total_cost:.4f} ({summary.total_calls} calls)")
        
        print(f"\n🧠 BY MODEL:")
        for model_name, summary in sorted(by_model.items()):
            print(f"   {model_name}: {summary.total_tokens:,} tokens, ${summary.total_cost:.4f} ({summary.total_calls} calls)")
        
        print(f"\n⚙️ BY OPERATION:")
        for operation, summary in sorted(by_operation.items()):
            print(f"   {operation}: {summary.total_tokens:,} tokens, ${summary.total_cost:.4f} ({summary.total_calls} calls)")
        
        print("="*60)
