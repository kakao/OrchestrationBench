"""
AWS Bedrock model implementation.

This module provides an interface to AWS Bedrock's models, specifically Anthropic Claude models.
"""

from typing import Dict, List, Any, Optional, Callable

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from loguru import logger

from src.models.base_model import BaseModel


class BedrockModel(BaseModel):
    """AWS Bedrock model implementation for Anthropic Claude models."""
    
    def __init__(
        self,
        model_name: str = "claude-3-haiku",
        temperature: float = 0.2,
        max_tokens: int = 10240,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        token_callback: Optional[Callable[[str, int, int, float], None]] = None,
        **kwargs
    ):
        """Initialize Bedrock model."""
        super().__init__(model_name, temperature, max_tokens, token_callback, **kwargs)
        self.client: Optional[boto3.client] = None
        self.region_name = aws_region or "us-east-1"
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        
        # Map common model names to Bedrock model IDs
        self.model_id_mapping = {
            "claude-3-haiku": "us.anthropic.claude-3-haiku-20240307-v1:0",
            "claude-3-sonnet": "us.anthropic.claude-3-sonnet-20240229-v1:0",
            "claude-3-opus": "us.anthropic.claude-3-opus-20240229-v1:0",
            "claude-3-5-sonnet": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "claude-3-7-haiku": "us.anthropic.claude-3-7-haiku-20240307-v1:0",
            "claude-sonnet-4": "us.anthropic.claude-sonnet-4-20250514-v1:0"
        }
        
        # Get the actual Bedrock model ID
        self.bedrock_model_id = self.model_id_mapping.get(model_name, model_name)
    
    async def initialize(self) -> None:
        """Initialize the Bedrock client."""
        try:
            # Prepare client configuration
            client_config = {
                "service_name": "bedrock-runtime",
                "region_name": self.region_name
            }
            
            # Add credentials if provided
            if self.aws_access_key_id and self.aws_secret_access_key:
                client_config.update({
                    "aws_access_key_id": self.aws_access_key_id,
                    "aws_secret_access_key": self.aws_secret_access_key
                })
                
                if self.aws_session_token:
                    client_config["aws_session_token"] = self.aws_session_token
            
            self.client = boto3.client(**client_config)
            
            # Test the connection
            await self.check_availability()
            self.initialized = True
            logger.info(f"Bedrock model {self.model_name} ({self.bedrock_model_id}) initialized successfully")
            
        except NoCredentialsError:
            logger.error("AWS credentials not found. Please configure your AWS credentials.")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock model: {e}")
            raise
    
    @staticmethod
    def parse_anthropic_response(response):
        try:
            output_message = response['output']['message']
            stop_reason = response['stopReason']
            content_list = output_message.get('content', [])
        except KeyError as e:
            raise ValueError(f"Invalid response format: missing key {e}")
        # 텍스트 내용 추출
        text_content = ""
        for content_item in content_list:
            if 'text' in content_item:
                text_content = content_item['text']
                break  # 첫 번째 텍스트 내용 사용
        
        # 도구 호출 처리
        tools = []
        if stop_reason == 'tool_use':
            for content_item in content_list:
                if 'toolUse' in content_item:
                    tool = content_item['toolUse']
                    tools.append({
                        "id": tool.get('toolUseId', ''),
                        "type": "function",
                        "function": {
                            "name": tool.get('name', ''),
                            "arguments": tool.get('input', {}),
                        }
                    })
        
        result= {
            "role": "assistant",
            "content": text_content
        }
        if tools:
            result["tool_calls"]= tools  
        return result

    @staticmethod
    def convert_to_anthropic_format(message: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a message to the format expected by Anthropic Claude."""
        content = []
        
        # Extract text content
        if isinstance(message.get('content'), str):
            content.append({'text': message['content']})
        elif isinstance(message.get('content'), list):
            for item in message['content']:
                if isinstance(item, str):
                    content.append({'text': item})
                elif isinstance(item, dict) and 'text' in item:
                    content.append({'text': item['text']})
        
        # Handle tool calls if present
        tool_calls = message.get('tool_calls', [])
        if tool_calls:
            for tool_call in tool_calls:
                content.append({
                    'toolUse': {
                        'toolUseId': tool_call['id'],
                        'name': tool_call['function']['name'],
                        'input': tool_call['function']['arguments']
                    }
                })
        
        return {
            'role': message['role'],
            'content': content
        }
    
    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate a response using Bedrock's converse API."""
        if not self.client:
            raise RuntimeError("Model not initialized")
        
        # Prepare messages
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        
        return await self.generate_chat_response(messages, system_prompt=system_prompt, **kwargs)
    
    async def generate_chat_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate a chat response using Bedrock's converse API."""
        if not self.client:
            raise RuntimeError("Model not initialized")
        
        # Check budget before making request
        if not self.check_budget():
            raise RuntimeError(f"Budget limit exceeded. Used: {self.budget_used}, Limit: {self.budget_limit}")
        
        # Convert messages to Bedrock format and extract system messages
        bedrock_messages = []
        extracted_system_prompt = system_prompt
        
        for msg in messages:
            msg_role = msg.get("role")
            
            # Handle system messages separately - Bedrock doesn't allow 'system' role in messages array
            if msg_role == "system":
                system_content = msg.get("content", "")
                if extracted_system_prompt:
                    # Combine with existing system prompt
                    extracted_system_prompt = f"{extracted_system_prompt}\n\n{system_content}"
                else:
                    extracted_system_prompt = system_content
                continue  # Skip adding to messages array
            
            # Only add user and assistant messages to Bedrock messages
            if msg_role in ["user", "assistant"]:
                bedrock_msg = self.convert_to_anthropic_format(msg)
                bedrock_messages.append(bedrock_msg)
        
        # Prepare inference configuration (AWS official example style)
        inference_config = {"temperature": self.temperature}
        
        # Additional model fields (AWS official example style)
        additional_model_fields = {"top_k": kwargs.get("top_k", 200)}
        
        tools = kwargs.get("tools", [])
        tool_config = self.convert_openai_tools_to_anthropic(tools)
        # Prepare system prompts in AWS format
        system_prompts = None
        if extracted_system_prompt:
            system_prompts = [{"text": extracted_system_prompt}]
        
        try:
            # Make the request using AWS official example pattern
            import asyncio
            loop = asyncio.get_event_loop()
            # Call converse method with AWS official parameters
            if system_prompts:
                if tools:
                    response = await loop.run_in_executor(
                        None,
                        lambda: self.client.converse(
                            modelId=self.bedrock_model_id,
                            messages=bedrock_messages,
                            toolConfig=tool_config,
                            system=system_prompts,
                            inferenceConfig=inference_config,
                            additionalModelRequestFields=additional_model_fields
                        )
                    )
                else:
                    response = await loop.run_in_executor(
                        None,
                        lambda: self.client.converse(
                            modelId=self.bedrock_model_id,
                            messages=bedrock_messages,
                            system=system_prompts,
                            inferenceConfig=inference_config,
                            additionalModelRequestFields=additional_model_fields
                        )
                    )
            else:
                if tools:
                    response = await loop.run_in_executor(
                        None,
                        lambda: self.client.converse(
                            modelId=self.bedrock_model_id,
                            messages=bedrock_messages,
                            toolConfig=tool_config,
                            inferenceConfig=inference_config,
                            additionalModelRequestFields=additional_model_fields
                        )
                    )
                else:
                    response = await loop.run_in_executor(
                        None,
                        lambda: self.client.converse(
                            modelId=self.bedrock_model_id,
                            messages=bedrock_messages,
                            inferenceConfig=inference_config,
                            additionalModelRequestFields=additional_model_fields
                        )
                    )
                
            converted_response = self.parse_anthropic_response(response)
            # Extract token usage information
            usage = response.get("usage", {})
            input_tokens = usage.get("inputTokens", 0)
            output_tokens = usage.get("outputTokens", 0)
            
            # Update statistics
            self._update_stats(input_tokens, output_tokens)
            
            # Log budget warning if approaching limit
            budget_info = self.get_budget_info()
            if budget_info.get("is_budget_warning"):
                logger.warning(f"Budget warning: {budget_info['percentage_used']}% used ({budget_info['budget_used']}/{budget_info['budget_limit']} tokens)")
            
            return converted_response
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            logger.error(f"Bedrock ClientError [{error_code}]: {error_message}")
            raise
        except Exception as e:
            logger.error(f"Error generating Bedrock response: {e}")
            raise
    
    async def check_availability(self) -> bool:
        """Check if Bedrock API is available."""
        if not self.client:
            return False
        
        try:
            # Make a simple test request using AWS official pattern
            import asyncio
            loop = asyncio.get_event_loop()
            
            test_messages = [{
                "role": "user",
                "content": [{"text": "Hello"}]
            }]
            
            # Use AWS official converse pattern
            response = await loop.run_in_executor(
                None,
                lambda: self.client.converse(
                    modelId=self.bedrock_model_id,
                    messages=test_messages,
                    inferenceConfig={"temperature": 0.5},
                    additionalModelRequestFields={"top_k": 200}
                )
            )
            return True
        except Exception as e:
            logger.warning(f"Bedrock availability check failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        base_info = super().get_model_info()
        base_info.update({
            "provider": "AWS Bedrock",
            "bedrock_model_id": self.bedrock_model_id,
            "region": self.region_name,
            "available_models": list(self.model_id_mapping.keys())
        })
        return base_info
    @staticmethod
    def convert_openai_tools_to_anthropic(openai_tools):
        """
        OpenAI 형태의 tools를 Anthropic 형태로 변환
        
        Args:
            openai_tools: OpenAI 형태의 tools 리스트
            예시: [
                {
                    "type": "function",
                    "function": {
                        "name": "top_song",
                        "description": "Get the most popular song...",
                        "parameters": {
                            "type": "object",
                            "properties": {...},
                            "required": [...]
                        }
                    }
                }
            ]
        
        Returns:
            Anthropic 형태의 tools 리스트
        """
        anthropic_tools = []
        
        for tool in openai_tools:
            if tool.get("type") == "function" and "function" in tool:
                func = tool["function"]
                
                anthropic_tool = {
                    "toolSpec": {
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "inputSchema": {
                            "json": func.get("parameters", {})
                        }
                    }
                }
                
                anthropic_tools.append(anthropic_tool)

        return {
            "tools": anthropic_tools
        }

    @staticmethod
    def convert_anthropic_tools_to_openai(anthropic_tools):
        """
        Anthropic 형태의 tools를 OpenAI 형태로 변환 (역변환)
        
        Args:
            anthropic_tools: Anthropic 형태의 tools 리스트
        
        Returns:
            OpenAI 형태의 tools 리스트
        """
        openai_tools = []
        
        for tool in anthropic_tools:
            if "toolSpec" in tool:
                spec = tool["toolSpec"]
                
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": spec.get("name", ""),
                        "description": spec.get("description", ""),
                        "parameters": spec.get("inputSchema", {}).get("json", {})
                    }
                }
                
                openai_tools.append(openai_tool)
        
        return openai_tools


    @staticmethod
    def create_anthropic_tools_request(openai_tools):
        """
        OpenAI tools를 받아서 완전한 Anthropic 요청 형태로 변환
        
        Args:
            openai_tools: OpenAI 형태의 tools 리스트
        
        Returns:
            Anthropic API 요청에 사용할 수 있는 딕셔너리
        """
        anthropic_tools = BedrockModel.convert_openai_tools_to_anthropic(openai_tools)

        return {
            "tools": anthropic_tools
        }

class BedrockClaudeHaiku(BedrockModel):
    """Bedrock Claude 3 Haiku model."""
    
    def __init__(self, **kwargs):
        super().__init__(model_name="claude-3-haiku", **kwargs)


class BedrockClaudeSonnet(BedrockModel):
    """Bedrock Claude 3 Sonnet model."""
    
    def __init__(self, **kwargs):
        super().__init__(model_name="claude-3-sonnet", **kwargs)


class BedrockClaude35Sonnet(BedrockModel):
    """Bedrock Claude 3.5 Sonnet model."""
    
    def __init__(self, **kwargs):
        super().__init__(model_name="claude-3-5-sonnet", **kwargs)


class BedrockClaude35Haiku(BedrockModel):
    """Bedrock Claude 3.5 Haiku model."""
    
    def __init__(self, **kwargs):
        super().__init__(model_name="claude-3-5-haiku", **kwargs)


class BedrockClaudeOpus(BedrockModel):
    """Bedrock Claude 3 Opus model."""
    
    def __init__(self, **kwargs):
        super().__init__(model_name="claude-3-opus", **kwargs)
