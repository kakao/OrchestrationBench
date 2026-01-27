import yaml
import json
from typing import Dict, List, Any, Optional

class ToolResultFetcher:
    def __init__(self, data: Dict):
        self.data = data
    
    def get_tool_result(self, tool_name: str, arguments_string: str, required_fields: List[str]) -> Optional[Dict[str, Any]]:
        """
        Returns the tool_call_result if all required_fields are satisfied by the tool call arguments.

        Args:
            tool_name: Name of the tool (e.g., "bookMovieTicket")
            arguments_string: JSON string of tool call arguments
            required_fields: List of required fields

        Returns:
            The 'data' part of tool_call_result as JSON if conditions are met, otherwise None.
        """
        try:
            # arguments_string을 JSON으로 파싱
            arguments = json.loads(arguments_string)
            
            # required_fields가 모두 채워져 있는지 확인
            if not self._check_required_fields(arguments, required_fields):
                return None
            
            # YAML에서 해당 tool의 결과 찾기
            tool_result = self._find_tool_result(tool_name, arguments)
            
            if tool_result:
                return tool_result
            else:
                return None
                
        except json.JSONDecodeError:
            print(f"Invalid JSON in arguments_string: {arguments_string}")
            return None
        except Exception as e:
            print(f"Error processing tool result: {e}")
            return None
    
    def _check_required_fields(self, arguments: Dict[str, Any], required_fields: List[str]) -> bool:
        """
        Checks if all required fields exist and are not empty.

        Args:
            arguments: Parsed arguments dictionary
            required_fields: List of required fields

        Returns:
            True if all required fields are present and filled, False otherwise
        """
        def _check_nested_field(obj: Any, field_path: str) -> bool:
            if '.' in field_path:
                # 중첩된 필드인 경우 (예: "userInfo.name")
                parts = field_path.split('.')
                current = obj
                for part in parts:
                    if not isinstance(current, dict) or part not in current:
                        return False
                    current = current[part]
                return current is not None and current != ""
            else:
                # 단순 필드인 경우
                return field_path in obj and obj[field_path] is not None and obj[field_path] != ""
        
        for field in required_fields:
            if not _check_nested_field(arguments, field):
                return False
        return True
    
    def _find_tool_result(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Finds and returns the result for the specified tool from the YAML data.

        Args:
            tool_name: Name of the tool
            arguments: Tool call arguments

        Returns:
            The 'data' part of tool_call_result if found, otherwise None.
        """        
        for i, step in enumerate(self.data):
            agent = step.get('agent', '')
            content = step.get('content', '')
            if content and f'tool: {tool_name}' in content:
                print(f"✅ Found tool '{tool_name}' in step {i}")
                print(f"🔍 Full content of step {i}:")
                print(f"'{content}'")
                # tool_call_result 부분 추출
                if 'tool_call_result:' in content:
                    print(f"✅ Found tool_call_result in step {i}")
                    try:
                        # tool_call_result 이후의 YAML 부분만 추출
                        result_start = content.find('tool_call_result:')
                        result_yaml = content[result_start:]
                        print(f"🔍 Parsing YAML: {result_yaml[:200]}...")
                        
                        # YAML 파싱
                        result_data = yaml.safe_load(result_yaml)
                        
                        if result_data and 'tool_call_result' in result_data:
                            result = result_data['tool_call_result'].get('data', {})
                            print(f"✅ Returning tool result: {result}")
                            return result
                        else:
                            print(f"❌ No 'tool_call_result' in parsed data: {result_data}")
                    except yaml.YAMLError as e:
                        print(f"❌ Error parsing YAML in step: {e}")
                        continue
                else:
                    print(f"❌ No 'tool_call_result:' found in content")
            else:
                print(f"❌ Tool '{tool_name}' not found in step {i}")
        
        print(f"❌ No matching tool result found for '{tool_name}'")
        return None

# 사용 예시
if __name__ == "__main__":
    # 사용 예시
    
    yaml_file_path="../../data/evaluation/2.yaml"
    with open(yaml_file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)

    fetcher = ToolResultFetcher(data)  # YAML 파일 경로
    
    # bookMovieTicket 예시
    arguments_string = '''
    {
        "refinedQuery": "7월 10일 18:00 CGV 강남 인사이드 아웃 2 예매",
        "movieTitle": "인사이드 아웃 2",
        "theaterName": "CGV 강남",
        "showDate": "2025-07-10",
        "showTime": "18:00",
        "audienceCount": 1,
        "audienceTypes": "adult",
        "userInfo": {
            "name": "박지나",
            "contactNumber": "010-1222-2222"
        }
    }
    '''
    
    required_fields = [
        "refinedQuery", "movieTitle", "theaterName", 
        "showDate", "showTime", "audienceCount", 
        "audienceTypes", "userInfo"
    ]
    
    result = fetcher.get_tool_result("bookMovieTicket", arguments_string, required_fields)
    
    if result:
        print("Tool call result found:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("Required fields not satisfied or result not found")