"""
Google Gemini model implementation.

This module provides an interface to Google's Gemini models using the new 'google.genai' SDK.
"""

from typing import Dict, List, Any, Optional, Callable
import copy
import google.genai as genai
from google.genai import Client
from google.genai import types # Corrected import for GenerationConfig, ChatSession, etc.
from loguru import logger

from src.models.base_model import BaseModel # Assuming BaseModel remains the same
from google.genai.types import GenerationConfig, Content, Part, Tool # Importing types from the new SDK

class GeminiModel(BaseModel):
    """Google Gemini model implementation using the new google.genai SDK."""
    
    def __init__(
        self,
        model_name: str = "gemini-1.5-pro", 
        temperature: float = 0.7,
        max_tokens: int = 4096,
        google_api_key: Optional[str] = None,
        token_callback: Optional[Callable[[str, int, int, float], None]] = None,
        **kwargs
    ):
        """Initialize Gemini model."""
        super().__init__(model_name, temperature, max_tokens, token_callback, **kwargs)
        self.client: Optional[Client] = None # The new Client object
        self.api_key = google_api_key
    
    async def initialize(self) -> None:
        """Initialize the Gemini client."""
        if not self.api_key:
            raise ValueError("Google API key not provided")
        
        try:
            self.client = genai.Client(api_key=self.api_key)
            # Test the connection using the client
            await self.check_availability()
            self.initialized = True
            logger.info(f"Gemini model {self.model_name} initialized successfully with new SDK")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model with new SDK: {e}")
            raise


    def clean_and_convert_to_declarations(self, raw_tools):
        """
        원래 툴 스키마를 Gemini API가 요구하는 함수 선언 리스트로 변환합니다.
        - `oneOf` 또는 복수 타입을 `STRING`으로 단순화하고 설명을 동적으로 조합합니다.
        - API에서 지원하지 않는 속성을 제거하고, `type`을 대문자로 변환합니다.
        """
        tools_copy = copy.deepcopy(raw_tools)
        
        declarations = []
        for raw_tool in tools_copy:
            function_def = raw_tool.get('function')
            if not function_def:
                continue

            parameters = function_def.get('parameters', {})
            properties = parameters.get('properties', {})

            # 💡 [핵심 수정 로직] 문제가 되는 프로퍼티들을 먼저 수정합니다.
            for prop_name, prop_value in properties.items():
                # Case 1: 'oneOf' 키가 명시적으로 있는 경우
                if 'oneOf' in prop_value:
                    descriptions = [
                        item.get('description', '') 
                        for item in prop_value.get('oneOf', []) 
                        if item.get('description')
                    ]
                    new_description = " 혹은 ".join(descriptions)
                    
                    properties[prop_name] = {
                        'type': 'string',
                        'description': new_description
                    }
                # Case 2: 'type'이 리스트인 경우 (['string', 'object'])
                elif isinstance(prop_value.get('type'), list):
                    original_desc = prop_value.get('description', '')
                    format_instruction = " (좌표는 'lat:위도,lng:경도' 형식의 문자열로 제공)"
                    
                    properties[prop_name] = {
                        'type': 'string',
                        'description': original_desc + format_instruction
                    }
                
                elif 'date' in prop_name.lower():
                    prop_value['format'] = 'date-time'

            # 재귀적으로 전체 스키마를 정리하고 타입을 대문자로 변환하는 함수
            def recursive_clean(schema_node):
                if isinstance(schema_node, list):
                    return [recursive_clean(item) for item in schema_node]

                if not isinstance(schema_node, dict):
                    return schema_node

                cleaned_node = {}
                for key, value in schema_node.items():
                    if key in ['additionalProperties', 'default'] and value in [False, None]:
                        continue
                    elif key == 'type' and isinstance(value, str):
                        cleaned_node[key] = value.upper()
                    else:
                        cleaned_node[key] = recursive_clean(value)

                # Gemini requires `items` only on ARRAY types. If a non-array schema
                # has `items`, coerce the outer type to ARRAY to match the intended
                # semantics of a list of items.
                if 'items' in cleaned_node:
                    outer_type = cleaned_node.get('type', 'ARRAY')
                    if outer_type != 'ARRAY':
                        cleaned_node['type'] = 'ARRAY'
                        cleaned_node.pop('properties', None)
                        cleaned_node.pop('required', None)
                return cleaned_node

            declarations.append(recursive_clean(function_def))
            
        return declarations
    
    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate a response using Gemini's API with the new SDK."""
        if not self.client:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        # Check budget before making request
        if not self.check_budget():
            raise RuntimeError(f"Budget limit exceeded. Used: {self.budget_used}, Limit: {self.budget_limit}")
        
        try:
            if 'tools' in kwargs:
                tools = kwargs['tools']
            else:
                tools = []
            # Configure generation parameters
            function_declarations_list = self.clean_and_convert_to_declarations(tools)
            converted_tools = Tool(function_declarations=function_declarations_list)

            config = types.GenerateContentConfig(tools=[converted_tools], 
                                                temperature=self.temperature,
                                                system_instruction=system_prompt if system_prompt else None
            )

            contents = [types.Part(text=prompt)]

            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )

            content = response.text
            return_format = {"role":"assistant", "content": content if content is not None else "\n\n"}
            if response.function_calls is not None:
                tool_calls = []
                for cnt, fn in enumerate(response.function_calls):
                    tool_calls.append(self.convert_function_call_to_tool_call(fn, cnt))
                return_format["tool_calls"] = tool_calls

            input_tokens = len(prompt.split()) * 1.3
            if system_prompt:
                input_tokens += len(system_prompt.split()) * 1.3
            output_tokens = len(content.split()) * 1.3
            self._update_stats(int(input_tokens), int(output_tokens))
            
            budget_info = self.get_budget_info()
            if budget_info.get("is_budget_warning"):
                logger.warning(f"Budget warning: {budget_info['percentage_used']}% used ({budget_info['budget_used']}/{budget_info['budget_limit']} tokens)")
            
            return return_format

        except Exception as e:
            logger.error(f"Error generating Gemini response with new SDK: {e}")
            raise
    
    def convert_function_call_to_tool_call(self, fc: dict or Any, index: int = 0) -> Dict:
        # Converts a single function_call to tool_call-style dict
        return {
            "id": f"call_generated_{index}",
            "type": "function",
            "function": {
                "name": getattr(fc, "name"),
                "arguments": getattr(fc, "args"),
            }
        }


    async def generate_chat_response(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Generate a chat response using Gemini's API with the new SDK."""
        if not self.client:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        # Check budget before making request
        if not self.check_budget():
            raise RuntimeError(f"Budget limit exceeded. Used: {self.budget_used}, Limit: {self.budget_limit}")
        
        try:
            # Convert messages to the new SDK's chat history format
            # For the new SDK, chat history is directly passed to generate_content
            # or managed by a ChatSession created via client.models.start_chat()
            
            # The structure for chat history in the new SDK:
            # List of types.Content objects, where each Content has a role and parts.
            history_contents = []
            system_prompt= ""
            for msg in messages: # All except the last message
                if msg["role"] == "system":
                    system_prompt = msg["content"] 
                else:
                    role = "user" if msg["role"] == "user" else "model"
                    history_contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
            
            if 'tools' in kwargs:
                tools = kwargs['tools']
            else:
                tools = []
            function_declarations_list = self.clean_and_convert_to_declarations(tools)
            tools = Tool(function_declarations=function_declarations_list)
            config = types.GenerateContentConfig(tools=[tools], 
                                                temperature=self.temperature,
                                                system_instruction=system_prompt if system_prompt else None
            )

            # Use async generate_content to avoid blocking the event loop
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=history_contents,
                config=config
            )

            content = response.text
            return_format = {"role":"assistant", "content": content if content is not None else "\n\n"}
            if response.function_calls is not None:
                tool_calls = []
                for cnt, fn in enumerate(response.function_calls):
                    tool_calls.append(self.convert_function_call_to_tool_call(fn, cnt))

                return_format["tool_calls"] = tool_calls

            usage_md = getattr(response, "usage_metadata", None)
            prompt_tokens = getattr(usage_md, "prompt_token_count", None) if usage_md else None
            candidate_tokens = getattr(usage_md, "candidates_token_count", None) if usage_md else None
            self._update_stats(int(prompt_tokens or 0), int(candidate_tokens or 0))
            
            budget_info = self.get_budget_info()
            if budget_info.get("is_budget_warning"):
                logger.warning(f"Budget warning: {budget_info['percentage_used']}% used ({budget_info['budget_used']}/{budget_info['budget_limit']} tokens)")

            return return_format

        except Exception as e:
            logger.error(f"Error generating Gemini chat response with new SDK: {e}")
            raise
    
    async def check_availability(self) -> bool:
        """Check if Gemini API is available using the new SDK."""
        if not self.client:
            return False
        
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=[types.Part(text="Hello")]
            )
            return response.text is not None
        except Exception as e:
            logger.warning(f"Gemini availability check failed with new SDK: {e}")
            return False
