"""
Value F1 Calculation Utilities

This module provides utilities for calculating F1 scores and evaluating argument matches
between predicted and actual function calls, including LLM-based semantic evaluation.
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Set, Tuple

import httpx
from loguru import logger
import traceback
from src.utils.model_factory import ModelFactory

# Type mapping for tool descriptions
TYPE_MATCH = {
    "string": str,
    "integer": int, 
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict
}

# =============================================================================
# Core F1 Score Calculation Functions
# =============================================================================

def calculate_f1_score(predicted_keys: Any, actual_keys: Any) -> Dict[str, float]:
    """
    Calculate F1 score for predicted and actual key sets or lists.
    
    Args:
        predicted_keys: Predicted keys (set or list)
        actual_keys: Actual keys (set or list)
        
    Returns:
        Dictionary containing precision, recall, and f1-score
    """
    def make_hashable(seq):
        return [tuple(x) if isinstance(x, list) else x for x in seq]

    # Convert to set if input is a list, ensuring hashable elements
    if isinstance(predicted_keys, list):
        predicted_keys = set(make_hashable(predicted_keys))
    if isinstance(actual_keys, list):
        actual_keys = set(make_hashable(actual_keys))

    if not actual_keys and not predicted_keys:
        return {"precision": 1.0, "recall": 1.0, "f1-score": 1.0}
    
    if not actual_keys:
        return {"precision": 0.0, "recall": 1.0, "f1-score": 0.0}
    
    if not predicted_keys:
        return {"precision": 1.0, "recall": 0.0, "f1-score": 0.0}
    
    true_positives = len(predicted_keys.intersection(actual_keys))
    false_positives = len(predicted_keys - actual_keys)
    false_negatives = len(actual_keys - predicted_keys)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {"precision": precision, "recall": recall, "f1-score": f1}


def extract_key_value_pairs(call_data: Dict) -> Set[str]:
    """
    Extract key:value pairs from function call data.
    
    Args:
        call_data: Function call data with arguments
        
    Returns:
        Set of key:value pair strings
    """
    pairs = set()
    for key, values in call_data['arguments'].items():
        if isinstance(values, list):
            for value in values:
                pairs.add(f"{key}:{str(value)}")
        else:
            pairs.add(f"{key}:{str(values)}")
    return pairs


def comprehensive_analysis(results):
    """
    Perform comprehensive confusion matrix analysis for rejection cases and FC quality.
    
    Args:
        results: Dictionary containing confusion matrix values
        
    Returns:
        Dictionary with detailed performance metrics
    """
    tpr = results.get("total_true_positive_reject", 0)
    tnr = results.get("total_true_negative_reject", 0) 
    fpr = results.get("total_false_positive_reject", 0)
    fnr = results.get("total_false_negative_reject", 0)
    rtm = results.get("total_rejection_type_mismatch", 0)  # rejection type mismatch

    # FC confusion matrix 메트릭들
    true_positive_fc = results.get("total_true_positive_fc", 0)  # 실제 FC, 예측 FC
    false_positive_fc = results.get("total_false_positive_fc", 0)  # 실제 reject, 예측 FC
    false_negative_fc = results.get("total_false_negative_fc", 0)  # 실제 FC, 예측 reject
    true_negative_fc = results.get("total_true_negative_fc", 0)  # 실제 reject, 예측 reject

    total_rejection_cases = results.get("total_rejection_cases", 0)
    total_function_call_cases = results.get("total_function_call_cases", 0)            

    # 이제 rejection_type_mismatch를 포함한 계산
    failed_cases = total_rejection_cases + total_function_call_cases - (tpr + tnr + fpr + fnr + rtm)

    # 1. 전체 의사결정 정확도
    total_decisions = tpr + tnr + fpr + fnr + rtm + failed_cases
    overall_accuracy = (tpr + tnr) / total_decisions if total_decisions > 0 else 0
    
    # 2. Reject 세부 분석
    overall_rejection_accuracy = tpr / total_rejection_cases if total_rejection_cases > 0 else 0
    # Rejection F1도 계산
    reject_precision = tpr / (tpr + fpr) if (tpr + fpr) > 0 else 0
    reject_recall = tpr / (tpr + fnr) if (tpr + fnr) > 0 else 0
    reject_f1 = 2 * (reject_precision * reject_recall) / (reject_precision + reject_recall) if (reject_precision + reject_recall) > 0 else 0
    reject_accuracy = (tpr + tnr) / (tpr + tnr + fpr + fnr) if (tpr + tnr + fpr + fnr) > 0 else 0

    # 3. FC 세부 분석 - 실제 FC confusion matrix 사용
    fc_precision = true_positive_fc / (true_positive_fc + false_positive_fc) if (true_positive_fc + false_positive_fc) > 0 else 0
    fc_recall = true_positive_fc / (true_positive_fc + false_negative_fc) if (true_positive_fc + false_negative_fc) > 0 else 0
    fc_f1 = 2 * (fc_precision * fc_recall) / (fc_precision + fc_recall) if (fc_precision + fc_recall) > 0 else 0
    
    # 4. 에러 패턴 분석
    total_errors = fpr + fnr + rtm  # rejection type mismatch도 에러로 간주
    overaction_ratio = fnr / total_errors if total_errors > 0 else 0  # 너무 많이 FC
    underaction_ratio = fpr / total_errors if total_errors > 0 else 0  # 너무 적게 FC
    type_mismatch_ratio = rtm / total_errors if total_errors > 0 else 0  # rejection 타입 틀림
    
    # 5. 추가: rejection type 정확도 분석
    total_rejection_predictions = tpr + rtm  # 실제로 rejection을 예측한 케이스들
    rejection_type_accuracy = tpr / total_rejection_predictions if total_rejection_predictions > 0 else 0
    
    # 6. FC 품질 분석
    fc_accuracy = (true_positive_fc + true_negative_fc) / (true_positive_fc + false_positive_fc + false_negative_fc + true_negative_fc) if (true_positive_fc + false_positive_fc + false_negative_fc + true_negative_fc) > 0 else 0
    
    return {
        "overall_decision_accuracy": overall_accuracy,
        "overall_rejection_accuracy": overall_rejection_accuracy,
        "reject_decision_performance": {
            "precision": reject_precision,
            "recall": reject_recall,
            "f1": reject_f1,
            "accuracy": reject_accuracy
        },
        "fc_decision_performance": {
            "precision": fc_precision,
            "recall": fc_recall,
            "f1": fc_f1,
            "accuracy": fc_accuracy  # FC 전체 정확도 추가
        },
        "error_patterns": {
            "overaction_rate": overaction_ratio,
            "underaction_rate": underaction_ratio,
            "type_mismatch_rate": type_mismatch_ratio  # rejection type 불일치 비율
        },
        "rejection_type_analysis": {
            "type_accuracy": rejection_type_accuracy,  # rejection 타입 정확도
            "type_mismatch_count": rtm,  # rejection 타입 불일치 개수
            "total_rejection_predictions": total_rejection_predictions  # 총 rejection 예측 개수
        },
        "fc_confusion_matrix": {
            "tp": true_positive_fc,  # True Positive
            "fp": false_positive_fc,  # False Positive
            "fn": false_negative_fc,  # False Negative
            "tn": true_negative_fc   # True Negative
        },
        "rejection_confidence": {
            "tp": tpr,  # True Positive
            "fp": fpr,  # False Positive
            "fn": fnr,  # False Negative
            "tn": tnr   # True Negative
        }

    }

# =============================================================================
# Tool Description Analysis Functions
# =============================================================================

def is_enum(tool_desc: Dict, key_name: str) -> bool:
    """Check if a key has enum constraints in tool descriptions."""
    for tool_name in tool_desc:
        properties = tool_desc[tool_name].get('properties', {})
        if key_name in properties:
            prop = properties[key_name]
            if 'enum' in prop or 'enum' in prop.get('items', {}):
                return True
    return False


def has_pattern(tool_desc: Dict, key_name: str) -> str:
    """Get regex pattern for a key from tool descriptions."""
    for tool_name in tool_desc:
        properties = tool_desc[tool_name].get('properties', {})
        if key_name in properties and 'pattern' in properties[key_name]:
            return properties[key_name]['pattern'].replace("\\\\", "\\")
    return None


def return_types(tool_desc: Dict, key_name: str) -> type:
    """Get expected type for a key from tool descriptions."""
    for tool_name in tool_desc:
        properties = tool_desc[tool_name].get('properties', {})
        if key_name in properties and 'type' in properties[key_name]:
            type_str = properties[key_name]['type']
            # Handle case where type is a list or other unhashable type
            if isinstance(type_str, list):
                # If it's a list, take the first type or convert to string
                type_str = type_str[0] if type_str else 'string'
            elif not isinstance(type_str, (str, int, float, bool, type(None))):
                # If it's not a hashable type, convert to string
                type_str = str(type_str)
            return TYPE_MATCH.get(type_str, type_str)
    return None

def should_skip_llm_check(fp_values: Set, fn_values: Set, key: str, tool_descriptions: Dict) -> Tuple[bool, str]:
    """
    Determine if LLM check is needed based on FP and FN values.
    
    Args:
        fp_values: False positive values
        fn_values: False negative values  
        key: Argument key name
        tool_descriptions: Tool description schemas
        
    Returns:
        Tuple of (should_skip, reason)
    """
    # 1. Check type consistency
    for value in fp_values:
        key_type = return_types(tool_descriptions, key)
        if key_type == int:
            try:
                int(value)
            except ValueError:
                return True, 'type_mismatch'
        elif key_type and not isinstance(value, key_type):
            return True, 'type_mismatch'

    # 2. Check enum constraints
    if len(fp_values) == len(fn_values) and is_enum(tool_descriptions, key):
        return True, 'enum_exact_comparison'

    # 3. Check pattern constraints
    pattern = has_pattern(tool_descriptions, key)
    if pattern is not None:
        for value in fp_values:
            if not re.match(pattern, str(value)):
                return True, 'pattern_mismatch'
    
    return False, None


# =============================================================================
# Argument Analysis Functions  
# =============================================================================

def analyze_argument_matches(p_call: Dict, a_call: Dict, tool_descriptions: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze matches between predicted and actual arguments with TP/FP/FN counts.
    
    Args:
        p_call: Predicted function call
        a_call: Actual function call
        tool_descriptions: Tool description schemas
        
    Returns:
        Dictionary containing auto_results and llm_needed lists
    """
    p_args = p_call['arguments']
    a_args = a_call['arguments']
    
    auto_results = []
    llm_needed = []
    all_keys = set(p_args.keys()) | set(a_args.keys())
    
    for key in all_keys:
        if key not in p_args:
            # Key missing in predicted → False Negative
            a_vals = a_args[key]
            auto_results.append({
                'key': key,
                'tp_count': 0,
                'fp_count': 0,
                'fn_count': len(a_vals),
                'source': 'auto',
                'reason': 'key_missing_in_predicted'
            })
            continue
            
        if key not in a_args:
            # Key missing in actual → False Positive  
            p_vals = p_args[key]
            auto_results.append({
                'key': key,
                'tp_count': 0,
                'fp_count': len(p_vals),
                'fn_count': 0,
                'source': 'auto',
                'reason': 'key_missing_in_actual'
            })
            continue
        
        p_vals = p_args[key]
        a_vals = a_args[key]

        # Convert to sets for comparison 
        def safe_sort_and_stringify(v):
            """Safely sort and stringify values, handling unhashable types."""
            def _safe_stringify(value):
                """Helper function to safely convert any value to string."""
                try:
                    if isinstance(value, (list, tuple)):
                        # For lists/tuples, recursively handle each element
                        elements = []
                        for element in value:
                            elements.append(_safe_stringify(element))
                        try:
                            # Try to sort if possible
                            return str(sorted(elements))
                        except (TypeError, ValueError):
                            # If sorting fails, just convert to string
                            return str(elements)
                    elif isinstance(value, dict):
                        # Sort dict items by key and recursively handle values
                        try:
                            sorted_items = []
                            for k in sorted(value.keys()):
                                sorted_items.append((str(k), _safe_stringify(value[k])))
                            return str(dict(sorted_items))
                        except (TypeError, ValueError):
                            # If sorting keys fails, use original order
                            items = [(str(k), _safe_stringify(val)) for k, val in value.items()]
                            return str(dict(items))
                    elif isinstance(value, set):
                        # Convert set to list and try to sort
                        try:
                            return str(sorted([_safe_stringify(item) for item in value]))
                        except (TypeError, ValueError):
                            return str([_safe_stringify(item) for item in value])
                    else:
                        return str(value)
                except Exception:
                    # Ultimate fallback
                    return str(value)
            
            try:
                return _safe_stringify(v)
            except Exception as e:
                # Final fallback: just convert to string
                logger.debug(f"Failed to sort and stringify value {type(v)}: {e}")
                return str(v)
        
        # Ensure p_vals and a_vals are lists for consistent processing
        if not isinstance(p_vals, list):
            p_vals = [p_vals]
        if not isinstance(a_vals, list):
            a_vals = [a_vals]
        
        try:
            p_set = set(safe_sort_and_stringify(v) for v in p_vals)
            a_set = set(safe_sort_and_stringify(v) for v in a_vals)
        except TypeError as e:
            # If we still can't create sets, fall back to simple string comparison
            logger.warning(f"Failed to create sets for key {key}: {e}")
            logger.warning(f"p_vals type: {type(p_vals)}, content: {p_vals}")
            logger.warning(f"a_vals type: {type(a_vals)}, content: {a_vals}")
            
            # Try with individual elements to see which one is problematic
            try:
                p_strings = []
                for i, v in enumerate(p_vals):
                    try:
                        p_strings.append(safe_sort_and_stringify(v))
                    except Exception as ve:
                        logger.warning(f"p_vals[{i}] failed: {type(v)} -> {ve}")
                        p_strings.append(str(v))
                p_set = set(p_strings)
            except Exception as pe:
                logger.warning(f"p_set creation failed completely: {pe}")
                p_set = {str(p_vals)}
            
            try:
                a_strings = []
                for i, v in enumerate(a_vals):
                    try:
                        a_strings.append(safe_sort_and_stringify(v))
                    except Exception as ve:
                        logger.warning(f"a_vals[{i}] failed: {type(v)} -> {ve}")
                        a_strings.append(str(v))
                a_set = set(a_strings)
            except Exception as ae:
                logger.warning(f"a_set creation failed completely: {ae}")
                a_set = {str(a_vals)}

        # Calculate automatic matches
        tp_values = p_set & a_set  # intersection (both have)
        fp_values = p_set - a_set  # predicted only
        fn_values = a_set - p_set  # actual only
        
        auto_tp = len(tp_values)
        auto_fp = len(fp_values)
        auto_fn = len(fn_values)
        
        # Check if LLM evaluation is needed
        if fp_values and fn_values:
            should_skip, reason = should_skip_llm_check(fp_values, fn_values, key, tool_descriptions)
            if should_skip:
                auto_results.append({
                    'key': key,
                    'tp_count': auto_tp,
                    'fp_count': auto_fp,
                    'fn_count': auto_fn,
                    'source': 'auto',
                    'reason': reason
                })
            else:
                # LLM check needed for semantic similarity
                llm_needed.append({
                    'key': key,
                    'confirmed_tp_count': auto_tp,
                    'predicted_values': list(fp_values),
                    'actual_values': list(fn_values)
                })
        else:
            # Clear case - no ambiguity
            auto_results.append({
                'key': key,
                'tp_count': auto_tp,
                'fp_count': auto_fp,
                'fn_count': auto_fn,
                'source': 'auto',
                'reason': 'exact_comparison'
            })
    
    logger.debug(f"Auto results: {auto_results}")
    logger.debug(f"LLM needed cases: {llm_needed}")
    
    return {
        'auto_results': auto_results,
        'llm_needed': llm_needed
    }


# =============================================================================
# LLM Judge Functions
# =============================================================================

def create_llm_judge_prompt(llm_needed: List[Dict]) -> Tuple[str, Dict]:
    """
    Create LLM judge prompt for semantic matching.
    
    Args:
        llm_needed: List of cases needing LLM evaluation
        
    Returns:
        Tuple of (prompt_string, number_mapping)
    """
    if not llm_needed:
        return "", {}
    
    prompt = "Calculate True Positive (TP), False Positive (FP), and False Negative (FN) counts for each arguments:\n\n"
    
    case_number = 1
    number_mapping = {}
    
    for item in llm_needed:
        logger.debug(f"Processing item: {item}")
        key = item['key']
        prompt += f"**Argument: {key}:**\n"
        prompt += f" Prediction: '{item['predicted_values']}'\n"
        prompt += f" Actual: '{item['actual_values']}'\n\n"
        
        number_mapping[case_number] = {
            'key': key,
            'predicted_values': item['predicted_values'],
            'actual_values': item['actual_values'],
        }
        case_number += 1

    return prompt, number_mapping


def apply_llm_judgments(llm_needed: List[Dict], llm_judge_results: Dict[int, int]) -> List[Dict]:
    """
    Apply LLM judgments and calculate final TP/FP/FN counts.
    
    Args:
        llm_needed: List of cases that needed LLM evaluation
        llm_judge_results: LLM judgment results {key_name: judgment_dict}
        
    Returns:
        List of processed results with TP/FP/FN counts
    """
    llm_results = []
    
    for item in llm_needed:
        key = item['key']
        judgment = llm_judge_results.get(key, {})
        if judgment:
            llm_results.append({
                'key': key,
                'tp_count': judgment.get('tp_count', 0),
                'fp_count': judgment.get('fp_count', 0),
                'fn_count': judgment.get('fn_count', 0),
                'source': 'llm_processed'
            })
        else:
            # LLM에서 결과가 없는 경우 기본값 사용
            llm_results.append({
                'key': key,
                'tp_count': 0,
                'fp_count': len(item.get('predicted_values', [])),
                'fn_count': len(item.get('actual_values', [])),
                'source': 'llm_fallback'
            })
        
    return llm_results


# =============================================================================
# Metrics Calculation Functions
# =============================================================================

def calculate_tp_fp_fn_metrics(all_results: List[Dict]) -> Dict[str, Any]:
    """
    Calculate metrics based on TP/FP/FN counts.
    
    Args:
        all_results: List of results with tp_count, fp_count, fn_count
        
    Returns:
        Dictionary containing precision, recall, f1, and all_results
    """
    tp = fp = fn = 0
    
    for result in all_results:
        tp += result['tp_count']
        fp += result['fp_count'] 
        fn += result['fn_count']
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'all_results': all_results
    }


def combine_analysis_with_llm_results(
    analysis: Dict[str, Any],
    llm_judge_results: Dict[int, int],
    number_mapping: Dict
) -> Dict[str, Any]:
    """
    Combine initial analysis results with LLM judgment results.
    
    Args:
        analysis: Result from analyze_argument_matches()
        llm_judge_results: LLM judgment results {case_number: judgment}
        number_mapping: Mapping from create_llm_judge_prompt()
    
    Returns:
        Final evaluation metrics
    """
    auto_results = analysis['auto_results']
    llm_needed = analysis['llm_needed']

    # Check for key mismatches
    llm_needed_keys = [item['key'] for item in llm_needed]
    llm_result_keys = list(llm_judge_results.keys())

    # Apply LLM judgments
    llm_results = apply_llm_judgments(llm_needed, llm_judge_results)
    
    # Combine all results
    all_results = auto_results + llm_results
    # Calculate final metrics
    final_metrics = calculate_tp_fp_fn_metrics(all_results)
    logger.debug(f"Final metrics: {final_metrics}")
    
    return {
        'metrics': final_metrics,
        'auto_results_count': len(auto_results),
        'llm_needed_count': len(llm_needed),
        'llm_judge_results': llm_judge_results
    }


# =============================================================================
# LLM API Functions
# =============================================================================

async def _call_llm_with_semaphore(
    messages: List[Dict], 
    model: ModelFactory, 
    gen_config: Dict,
    max_retries: int = 3,
    max_concurrent: int = 5
) -> Dict:
    """Call LLM with concurrency control via semaphore."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    logger.debug(f"LLM messages: {messages}")
    async with semaphore:
        return await _call_llm(messages, model, gen_config, max_retries)


async def _call_llm(
    messages: List[Dict], 
    model: ModelFactory, 
    gen_config: Dict, 
    max_retries: int = 3
) -> Dict:
    """Make API call to LLM with retry logic using ModelFactory."""
        
    for attempt in range(max_retries):
        try:            
            # Call the model
            response = await model.generate_chat_response(messages, **gen_config)
        
            # Convert response to expected format
            if hasattr(response, 'content'):
                return {"choices": [{"message": {"content": response.content}}]}
            elif isinstance(response, str):
                return {"choices": [{"message": {"content": response}}]}
            else:
                return response
                
        except Exception as e:
            traceback.print_exc()
            logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                logger.error("Max retries reached for LLM call")
                return {"error": str(e)}
            await asyncio.sleep(2 ** attempt)
    
    return {"error": "Max retries exceeded"}


# =============================================================================
# Data Processing Utility Functions
# =============================================================================

def get_values_before_key(data: Dict, target_key: str) -> List[Dict[str, Any]]:
    """
    Returns all values before a specific key.
    
    Args:
        data: JSON data (dictionary)
        target_key: The key to stop at (e.g., "4-2")

    Returns:
        A list of key-value pairs
    """
    label = data.get("label", {})
    result = []
    
    def parse_key(key: str) -> List[int]:
        """Parse the key into a sortable format."""
        parts = key.split('-')
        return [int(part) for part in parts]

    # Sort the keys
    sorted_keys = sorted(label.keys(), key=parse_key)

    # Collect all values before target_key
    for key in sorted_keys:
        if key == target_key:
            break
        result.append({
            "key": key,
            "value": label[key]
        })
    
    return result


def create_history(data: Dict, target_key: str) -> List[Dict[str, str]]:
    """
    Generate conversation history from JSON data.
    - Includes only items with "user" or "assistant" roles
    - Excludes items with keys containing "-"
        
    Args:
        data: JSON data (dictionary)
        target_key: The key to stop at (e.g., "4-2")
    
    Returns:
        A list of dictionaries in the format [{"role": "...", "content": "..."}]
    """
    values_before_key = get_values_before_key(data, target_key)
    history = []
    
    for item in values_before_key:
        key = item["key"]
        value = item["value"]
        
        # Handle special key patterns
        if "-1" in key:
            prev_key = key.split("-")[0]
            if history and history[-1]["key"] == prev_key:
                history.pop(-1)
            if item["value"].get("tool_calls"):
                refined_queries = []
                for tool_call in item["value"]["tool_calls"]:
                    arguments = tool_call["function"]["arguments"]
                    arguments = json.loads(arguments)
                    if "refinedQuery" in arguments:
                        refined_queries.append(arguments["refinedQuery"])
                history_item = {
                    "key": "gen_user",
                    "role": "user",
                    "content": " ".join(refined_queries)
                }
                history.append(history_item)
        
        if "-2" in key:
            continue
            
        # Check if value is a dictionary with "role" and "content"
        if isinstance(value, dict) and "role" in value and "content" in value:
            role = value.get("role", "")
            content = value.get("content", "")
            # Only include "user" and "assistant" roles
            if role in ["user", "assistant"]:
                history_item = {
                    "key": key,
                    "role": role,
                    "content": content
                }
                history.append(history_item)
    
    # Filter out empty content
    history = [{"role": x["role"], "content": x["content"]} 
               for x in history if x["content"] != ""]
    return history

def try_to_parse_think_content( content: str) -> Dict[str, Any]:
    content = content.strip().split("</think>")[-1]
    json_matches = re.findall(r'\{.*\}', content, re.DOTALL| re.MULTILINE)
    if json_matches:
        json_str = json_matches[-1]
        # Try to parse as-is first
        try:
            parsed_result = json.loads(json_str)
            function_name = parsed_result.get("function_name", "")
            arguments = parsed_result.get("arguments", {})
            parsed_result ={
                "function_name": [function_name] if isinstance(function_name, str) and function_name!="" else function_name,
                "arguments": {key: [value] if isinstance(value, str) else value for key, value in arguments.items()}
            }
            return parsed_result
        except:
            return None
    return None

# =============================================================================
# Utility Functions for LLM Response Processing
# =============================================================================

def extract_content_from_llm_result(llm_result: Any) -> str:
    """
    Extract content from various LLM response formats.
    
    Args:
        llm_result: LLM response in various formats
        
    Returns:
        Extracted content string
    """
    content = ""
    if isinstance(llm_result, dict):
        if 'choices' in llm_result and len(llm_result['choices']) > 0:
            # Standard OpenAI-like response format
            choice = llm_result['choices'][0]
            if 'message' in choice:
                content = choice['message'].get('content', '')
                if not content.strip():
                    content = choice['message'].get('reasoning_content', '')
            else:
                content = str(choice)
        elif 'content' in llm_result:
            content = llm_result['content']
        else:
            content = str(llm_result)
    else:
        content = str(llm_result)
    
    return content


def is_rejection_case(call: Dict, call_type: str) -> bool:
    """
    Check if this is a case where function calling should be rejected.

    Args:
        call: Function call data
        call_type: Either "label" or "prediction"

    Returns:
        True if this is a rejection case
    """
    if call_type == "label":
        # Label: function_name이 빈 문자열이면 rejection
        return call["function_name"] == ""
    else:
        # Prediction: arguments에 명시적으로 rejection 패턴이 있어야 rejection
        args = call.get("arguments", {})
        return ("TOOL_CONSTRAINT_VIOLATION" in args or "AWAITING_USER_INPUT" in args)


def is_failed_case(call: Dict) -> bool:
    """
    Check if this is a failed function call case.
    
    Args:
        call: Function call data
        
    Returns:
        True if this is a failed case
    """
    return not "arguments" in call or (not isinstance(call["arguments"], dict) and
            "TOOL_CONSTRAINT_VIOLATION" not in call["arguments"] and 
            "AWAITING_USER_INPUT" not in call["arguments"])


def save_llm_results(input_data: List[Dict], llm_responses: List[Dict], output_dir: str, file_name: str) -> None:
    """
    Save LLM evaluation input and output data for analysis.
    
    Args:
        input_data: Input data sent to LLM
        llm_responses: Responses from LLM
        output_dir: Directory to save files
        file_name: Name of the output file
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    save_data = {
        'input_data': input_data,
        'llm_responses': llm_responses,
        'timestamp': str(asyncio.get_event_loop().time())
    }
    
    file_path = os.path.join(output_dir, f"{file_name}.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)


def parse_arguments_to_keys(arguments_str: str) -> Set[str]:
    """
    Parse arguments JSON string and extract keys.
    
    Args:
        arguments_str: Arguments as string or dict
        
    Returns:
        Set of argument keys
    """
    try:
        # Check if arguments_str is already a dict
        if isinstance(arguments_str, dict):
            return set(arguments_str.keys())
        
        # If it's a string, try to parse as JSON
        if isinstance(arguments_str, str):
            args_dict = json.loads(arguments_str)
            return set(args_dict.keys())
        
        # If it's neither dict nor string, return empty set
        return set()
    except json.JSONDecodeError:
        return set()
    except (TypeError, AttributeError):
        # Handle case where arguments_str is None or other unexpected type
        return set()


def _extract_actual_call(step_id: str, label_history: Dict, total_label_count: List[int]) -> Tuple[Dict, List[str]]:
    """Extract actual call from label history for given step_id."""
    # Determine label key
    label_key = f"{step_id}-1" if f"{step_id}-1" in label_history else step_id
    
    if label_key not in label_history:
        return {"function_name": "", "arguments": "no label found"}, []
    
    label_call = label_history[label_key]
    
    # Handle non-tool calls - rejection case
    if "tool_calls" not in label_call:
        actual_call = {
            "function_name": "",
            "arguments": label_call.get("content", "should be rejected function calling")
        }
        return actual_call, []
    
    # Process tool calls
    actual_aggregated_arguments = {}
    actual_function_names = []
    
    for single_tool_call in label_call["tool_calls"]:
        total_label_count[0] += 1
        function_name = single_tool_call.get("function", {}).get("name", "")
        arguments = single_tool_call.get("function", {}).get("arguments", {})
        _process_tool_call_arguments(arguments, actual_aggregated_arguments)
        actual_function_names.append(function_name)
    
    actual_call = {
        "function_name": actual_function_names,
        "arguments": actual_aggregated_arguments
    }
    return actual_call, actual_function_names

def _extract_predicted_call(one_step: Dict) -> Dict:
    """Extract predicted call from one_step."""
    # Handle tool calls
    if len(one_step.get("tool_calls", [])) == 0:
        # Check if this is an AWAITING_USER_INPUT case (rejection case)
        content = one_step.get("content", "")
        if "AWAITING_USER_INPUT" in content:
            return {
                "function_name": "rejection",  # Empty function name indicates rejection
                "arguments": {"AWAITING_USER_INPUT": "true"}
            }
        elif "TOOL_CONSTRAINT_VIOLATION" in content:
            return {
                "function_name": "rejection",  # Empty function name indicates rejection
                "arguments": {"TOOL_CONSTRAINT_VIOLATION": "true"}
            }
        else:
            # Empty tool_calls but no clear rejection signal - might be a mistake
            # Return a "no function call" result instead of None to allow comparison
            return {
                "function_name": "",  # Empty function name
                "arguments": {}  # Empty arguments - this won't match AWAITING_USER_INPUT pattern
            }
    else:        
        predicted_aggregated_arguments = {}
        predicted_function_names = []

        for single_tool_call in one_step["tool_calls"]:
            function_name = single_tool_call["function"]["name"]
            arguments = single_tool_call["function"]["arguments"]
            _process_tool_call_arguments(arguments, predicted_aggregated_arguments)
            predicted_function_names.append(function_name)
    
        if predicted_function_names:
            return {
                "function_name": predicted_function_names,
                "arguments": predicted_aggregated_arguments
            }
        
        # Handle content-based calls
        else:
            content = one_step.get("content", "")
            parsed_content = try_to_parse_think_content(content)
            if parsed_content is not None:
                return parsed_content
        
    return None

def calculate_total_rejection_cases(label_history: Dict) -> int:
    """
    Calculate total rejection cases using our manual counting logic.
    A rejection case is when agent_id exists in a label step but there's no next step.
    
    Args:
        label_history: Dictionary of label data
        
    Returns:
        Total number of rejection cases
    """
    rejection_count = 0
    total_label_fc_cases = 0
    # Look through all label keys for agent_id
    for label_key, label_content in label_history.items():
        if '-' in label_key:
            continue
        # Check if this label entry has agent_id
        if isinstance(label_content, dict) and 'agent_id' in label_content:
            # This is an agent step, check if the next step (key-1) exists
            next_key = f"{label_key}-1"
            # If the next step doesn't exist, it's a rejection case
            if next_key not in label_history:
                rejection_count += 1
            else:
                total_label_fc_cases += 1
    return rejection_count, total_label_fc_cases


def extract_both_tool_calls(sub_agent_history: List[Dict], 
                          label_history: Dict) -> Tuple[List[Dict], List[Dict], List[str], int, int, int, int, int, int, int, int, int, int]:
    """
    Extract tool calls from both sub_agent_history and label_history in one go.
    Also calculate rejection-related type1/type2 errors and FC quality metrics.
    
    Returns:
        tuple: (predicted_calls, actual_calls, step_list, 
               type1_errors, type2_errors, tp_reject, tn_reject,
               true_positive_fc, false_positive_fc, false_negative_fc, true_negative_fc,
               total_label_reject_cases, total_label_fc_cases)
    """
    predicted_calls = []
    actual_calls = []
    step_list = []
    
    # Confusion Matrix 구성 요소들 - 세분화된 rejection type 고려
    true_positive_reject = 0   # 실제 reject, 예측도 reject (같은 타입)
    true_negative_reject = 0   # 실제 FC, 예측도 FC  
    false_positive_reject = 0  # 실제 FC, 예측은 reject (Type2 error) + 실제 reject이지만 타입 틀림
    false_negative_reject = 0  # 실제 reject, 예측은 FC (Type1 error)
    
    # FC 품질 측정 메트릭들 (완전한 confusion matrix)
    true_positive_fc = 0                  # 실제 FC, 예측 FC (True Positive)
    false_positive_fc = 0                  # 실제 reject, 예측 FC (False Positive)
    false_negative_fc = 0                  # 실제 FC, 예측 reject (False Negative)
    true_negative_fc = 0                  # 실제 reject, 예측 reject (True Negative)

    total_label_reject_cases = 0
    total_label_fc_cases = 0
    rejection_type_mismatch = 0  # rejection 타입 불일치 카운트
    failed_generation = 0  # 예측 생성 실패 (FAILED)

    def get_rejection_type(call: Dict, call_type: str) -> str:
        """Get specific rejection type from call"""
        # FAILED 케이스 먼저 체크 (prediction만 해당)
        if call_type == "prediction" and is_failed_case(call):
            return "FAILED"

        if not is_rejection_case(call, call_type):
            return "FC"  # Function Call

        args = call.get("arguments", {})
        # arguments가 문자열인 경우 (label content)
        if isinstance(args, str):
            if "AWAITING_USER_INPUT" in args:
                return "AWAITING_USER_INPUT"
            elif "TOOL_CONSTRAINT_VIOLATION" in args:
                return "TOOL_CONSTRAINT_VIOLATION"
            else:
                return "GENERAL_REJECTION"
        # arguments가 딕셔너리인 경우
        if "AWAITING_USER_INPUT" in args:
            return "AWAITING_USER_INPUT"
        elif "TOOL_CONSTRAINT_VIOLATION" in args:
            return "TOOL_CONSTRAINT_VIOLATION"
        else:
            return "GENERAL_REJECTION"

    # Calculate total rejection cases using separate function
    for one_step in sub_agent_history:
        step_id = one_step['step_id']
        # Extract actual and predicted calls
        actual_call, actual_function_names = _extract_actual_call(
            step_id, label_history, [0]  # dummy counter for compatibility
        )
        predicted_call = _extract_predicted_call(one_step)

        # Rejection 타입 구분
        actual_type = get_rejection_type(actual_call, "label")
        predicted_type = get_rejection_type(predicted_call, "prediction") if predicted_call is not None else "FAILED"

        # Confusion Matrix 계산 - rejection 타입 일치 여부 고려
        # FAILED는 별도로 트래킹 (FP_reject에 포함시키지 않음)
        if actual_type == "FC":  # 실제 FC case
            if predicted_type == "FC":
                true_negative_reject += 1  # 정확히 FC함
                true_positive_fc += 1
            elif predicted_type == "FAILED":  # 생성 실패
                failed_generation += 1
                false_negative_fc += 1  # FC해야 하는데 못함
            else:  # predicted는 rejection 계열 (AWAITING_USER_INPUT, TOOL_CONSTRAINT_VIOLATION 등)
                false_positive_reject += 1  # Type2 error: FC해야 하는데 거부함
                false_negative_fc += 1
        else:  # 실제는 rejection 계열
            if predicted_type == "FC":
                false_negative_reject += 1  # Type1 error: 거부해야 하는데 FC함
                false_positive_fc += 1
            elif predicted_type == "FAILED":  # 생성 실패
                failed_generation += 1
                true_negative_fc += 1  # FC 관점에서는 TN (잘못 FC하진 않음)
            elif predicted_type == actual_type:
                true_positive_reject += 1  # 정확히 같은 타입으로 거부
                true_negative_fc += 1
            else:  # 다른 rejection 타입 (타입 불일치)
                rejection_type_mismatch += 1  # 타입 불일치
                true_positive_reject += 1  # reject은 맞았으니까 TP (타입만 틀림)
                true_negative_fc += 1  # FC 관점에서는 TN
        
        # Add to results if we have a predicted call
        if predicted_call is not None:
            predicted_calls.append(predicted_call)
            actual_calls.append(actual_call)
            step_list.append(step_id)
    
    total_label_reject_cases, total_label_fc_cases = calculate_total_rejection_cases(label_history)
    # FAILED는 별도 카운트, rejection_type_mismatch는 false_positive_reject에 포함
    assert (true_positive_reject + true_negative_reject +
            false_positive_reject + false_negative_reject + failed_generation ==
            len(sub_agent_history)), "Confusion matrix 계산 오류"

    return (predicted_calls, actual_calls, step_list,
            false_negative_reject,  # Type1 errors
            false_positive_reject,  # Type2 errors
            true_positive_reject,
            true_negative_reject,
            true_positive_fc,  # FC True Positive (실제 FC, 예측 FC)
            false_positive_fc,  # FC False Positive (실제 reject, 예측 FC)
            false_negative_fc,  # FC False Negative (실제 FC, 예측 reject)
            true_negative_fc,  # FC True Negative (실제 reject, 예측 reject)
            total_label_reject_cases,
            total_label_fc_cases,
            rejection_type_mismatch,  # Rejection 타입 불일치 카운트
            failed_generation)  # 예측 생성 실패 카운트

# 추가: Confusion Matrix 분석을 위한 헬퍼 함수
def print_confusion_matrix_stats(type1_errors, type2_errors, 
                               total_label_reject, total_label_fc):
    """
    Confusion Matrix 통계 출력
    """
    true_positive = total_label_reject - type1_errors  # 실제 reject, 예측 reject
    true_negative = total_label_fc - type2_errors      # 실제 FC, 예측 FC
    false_positive = type2_errors                      # 실제 FC, 예측 reject
    false_negative = type1_errors                      # 실제 reject, 예측 FC
    
    total = true_positive + true_negative + false_positive + false_negative
    
    print("Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                Reject  FC")
    print(f"Actual Reject    {true_positive:4d}  {false_negative:4d}")
    print(f"       FC        {false_positive:4d}  {true_negative:4d}")
    print()
    print(f"Accuracy: {(true_positive + true_negative) / total:.3f}")
    print(f"Precision (Reject): {true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0:.3f}")
    print(f"Recall (Reject): {true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0:.3f}")
    print(f"Type1 Error Rate: {false_negative / total:.3f}")
    print(f"Type2 Error Rate: {false_positive / total:.3f}")

def _process_tool_call_arguments(arguments, aggregated_arguments: dict):
    """
    Common function to process tool call arguments and aggregate them.
    """
    if not isinstance(arguments, dict):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            aggregated_arguments["error"] = f"Invalid JSON in arguments: {arguments}"
            return
    
    for key, val in arguments.items():
        if key in aggregated_arguments:
            aggregated_arguments[key].append(val)
        else:
            aggregated_arguments[key] = [val]

# =============================================================================
# Utility for Saving
# =============================================================================


def save_comprehensive_evaluation_results(results: Dict[str, Any], output_dir: str, file_identifier: str) -> None:
    """
    Save comprehensive evaluation results including workflow, key/value scores, and LLM results.
    
    Args:
        results: Complete evaluation results
        output_dir: Directory to save results
        file_identifier: Identifier for the file
    """
    import os
    from datetime import datetime
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 상세 분석 수행
    detailed_analysis = comprehensive_analysis(results)
    
    # Prepare comprehensive results
    comprehensive_results = {
        "file_identifier": file_identifier,
        "timestamp": datetime.now().isoformat(),
        "evaluation_summary": {
            "workflow_score_with_failure": results.get("workflow_evaluation_with_failure", 0.0),
            "workflow_score_without_failure": results.get("workflow_evaluation_without_failure", 0.0),
            "workflow_score": results.get("workflow_evaluation", 0.0),
            "agents_score": results.get("agents_evaluation", 0.0),
            "key_f1_score": results.get("arguments_key_score", 0.0),
            "value_f1_score": results.get("arguments_value_score_not_count_rejection", 0.0),
            "function_name_f1_score": results.get("function_name_accuracy", 0.0),
            "function_call_accuracy": results.get("total_fc_successful_calls", 0) / max(results.get("total_function_call_cases", 1), 1),
            # 새로운 상세 분석 결과 추가
            "overall_decision_accuracy": detailed_analysis["overall_decision_accuracy"],
            "reject_f1_score": detailed_analysis["reject_decision_performance"]["f1"],
            "fc_f1_score": detailed_analysis["fc_decision_performance"]["f1"],
            "overaction_error_rate": detailed_analysis["error_patterns"]["overaction_rate"],
            "underaction_error_rate": detailed_analysis["error_patterns"]["underaction_rate"]
        },
        "detailed_metrics": {
            "comprehensive_analysis": detailed_analysis,
            "workflow_evaluation": {
                "total_score_with_failure": results.get("workflow_evaluation_with_failure", 0.0),
                "total_score_without_failure": results.get("workflow_evaluation_without_failure", 0.0),
                "average_workflow_score": results.get("workflow_evaluation", 0.0),
                "average_agents_scores": results.get("agents_evaluation", 0.0),
                "failed_workflow_generation": results.get("failed_workflow_generation", 0),
                "total_comparison_steps_evaluated": results.get("total_comparison_steps_evaluated", 0)
            },
            "arguments_evaluation": {
                "key_score": results.get("detail", {}).get("arguments_detail", {}).get("key_score", {}),
                "value_score": results.get("detail", {}).get("arguments_detail", {}).get("value_score_not_count_rejection", {}),
                "function_name_score": results.get("detail", {}).get("arguments_detail", {}).get("function_name_score", {}),
                "function_call_metrics": {
                    "total_function_call_cases": results.get("total_function_call_cases", 0),
                    "total_fc_successful_calls": results.get("total_fc_successful_calls", 0),
                    "function_call_accuracy": results.get("total_fc_successful_calls", 0) / max(results.get("total_function_call_cases", 1), 1)
                },
                "llm_failed_trials": results.get("detail", {}).get("arguments_detail", {}).get("llm_failed_trials", [])
            }
        },
        "raw_confusion_matrix": {
            "total_rejection_cases": results.get("total_rejection_cases", 0), # calcuated from labels
            "total_function_call_cases": results.get("total_function_call_cases", 0), # calcuated from labels
            "total_true_positive_reject": results.get("total_true_positive_reject", 0),
            "total_true_negative_reject": results.get("total_true_negative_reject", 0),
            "total_false_positive_reject": results.get("total_false_positive_reject", 0),
            "total_false_negative_reject": results.get("total_false_negative_reject", 0),
        },
        "raw_workflow_detail": results.get("detail", {}).get("workflow_detail", {}),
        "raw_arguments_detail": results.get("detail", {}).get("arguments_detail", {}),
        "complete_raw_results": results  # Include complete raw results for reference
    }
    
    # Save to file
    output_path = os.path.join(output_dir, f"{file_identifier}_comprehensive_evaluation.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Comprehensive evaluation results saved: {output_path}")
