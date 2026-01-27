import re
import json
import yaml
import asyncio
import os
import glob
from datetime import datetime
import traceback
from loguru import logger
from typing import Dict, List, Set, Any
from src.utils.evaluation.eval_utils import (
    analyze_argument_matches,
    create_llm_judge_prompt,
    combine_analysis_with_llm_results,
    calculate_f1_score,
    create_history,
    extract_content_from_llm_result,
    _call_llm_with_semaphore,
    parse_arguments_to_keys,
    extract_both_tool_calls,
    _process_tool_call_arguments,
    save_llm_results,
    is_failed_case,
    is_rejection_case
)
from src.utils.model_factory import ModelFactory

def process_single_llm_result(llm_result, metadata, call_index):
    """Process individual LLM result and return evaluation metrics"""
    try:
        # Parse LLM judgment results
        llm_judge_results = {}
        if llm_result and not isinstance(llm_result, Exception):
            try:
                # Handle different response structures
                content = extract_content_from_llm_result(llm_result)                
                # Extract JSON from content - try multiple patterns
                json_str = ""
                
                # Pattern 1: Look for {key: value} format
                content = content.strip().split("</think>")[-1]
                json_matches = re.findall(r'\{.*\}', content, re.DOTALL| re.MULTILINE)
                if json_matches:
                    json_str = json_matches[-1]
                    # Try to parse as-is first
                    try:
                        llm_judge_results = json.loads(json_str)
                    except json.JSONDecodeError as e:
                        fixed_json = json_str.replace("'", '"')
                        try:
                            llm_judge_results = json.loads(fixed_json)
                        except json.JSONDecodeError as e2:
                            llm_judge_results = {}
                else:
                    print(f"No JSON matches found in content")
                
            except Exception as e:
                print(f"Exception while parsing LLM response: {e}")
                llm_judge_results = {}
        
        elif isinstance(llm_result, Exception):
            logger.error(f"LLM call failed for call {call_index}: {llm_result}")
            llm_judge_results = {}
        
        analysis = {
            'auto_results': metadata['auto_results'],
            'llm_needed': metadata['llm_needed']
        }

        final_result = combine_analysis_with_llm_results(analysis, llm_judge_results, metadata['number_mapping'])
        result = {
            'call_index': call_index,
            'precision': final_result['metrics']['precision'],
            'recall': final_result['metrics']['recall'],
            'f1': final_result['metrics']['f1']
        }

        # Create eval_data for logging
        eval_data = {
            'call_index': call_index, 
            'p_call': metadata['p_call'], 
            'a_call': metadata['a_call'], 
            'auto_results': metadata['auto_results'], 
            'llm_needed': metadata['llm_needed'], 
            'number_mapping': metadata['number_mapping'], 
            'messages': metadata['history']
        }   
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing call {call_index}: {e}")
        return {
            'call_index': call_index,
            'error': str(e),
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0
        }

async def call_evaluation_llm(predicted_calls: List[Dict],
                                actual_calls: List[Dict],
                                eval_config : Dict,
                                judge_model: ModelFactory,
                                system_info: str = "",
                                llm_history: List[Dict] = None,
                                tool_descriptions: Dict[str, Any] = None,
                                save_results: bool = False,
                                output_dir: str = None,
                                file_identifier: str = None) -> List[Dict]:
    """
    Call evaluation LLM for argument comparison with concurrency control and batch processing.
    Optionally save input/output data for analysis.
    """
    skip_llm_eval = eval_config.get("skip_llm_eval", False)
    max_concurrent = eval_config.get("max_concurrent", 3)  # Reduced for better order preservation
    batch_size = eval_config.get("batch_size", 8)  # Smaller batches for better matching
    
    # Prepare data for saving if requested
    input_data = []
    llm_responses = []
        
    if not predicted_calls or not actual_calls:
        return []
    
    gen_config = eval_config["judge"]["generation_params"]
    model_config = eval_config["judge"]["model"]
    
    total_calls = len(predicted_calls)
    all_results = []    
    
    # Collect all LLM requests for batch processing
    batch_requests = []
    batch_metadata = []
    
    # First pass: prepare all requests
    for global_idx in range(len(predicted_calls)):
        p_call = predicted_calls[global_idx]
        a_call = actual_calls[global_idx]
        history = llm_history[global_idx] if llm_history else None
        call_index = global_idx
        applied_tool_desc = {function_name: tool_descriptions[function_name] for function_name in p_call['function_name'] if function_name in tool_descriptions}

        if not is_rejection_case(a_call, 'label') and not is_rejection_case(p_call, 'prediction') and not is_failed_case(p_call):
            analysis = analyze_argument_matches(p_call, a_call, applied_tool_desc)
            auto_results = analysis['auto_results']
            llm_needed = analysis['llm_needed']
            
            if llm_needed and not skip_llm_eval:
                llm_prompt, number_mapping = create_llm_judge_prompt(llm_needed)
                system_prompt = eval_config["prompts"]["arguments_evaluation"]["prompt"]
                system_prompt = system_prompt.replace("%%system_info%%", system_info)
                system_prompt = system_prompt.replace("%%tool_description%%", json.dumps(applied_tool_desc, indent=2, ensure_ascii=False))
                if history:
                    history.insert(0, {"role": "system", "content": system_prompt})
                else:
                    history = [{"role": "system", "content": system_prompt}]
                llm_prompt = "</history>\n\n" + llm_prompt
                history = history + [{"role": "user", "content": llm_prompt}]
                
                # Add unique identifier to the prompt for batch processing
                batch_id = f"EVAL_BATCH_{call_index}_{global_idx}"
                history_with_id = history.copy()
                history_with_id[-1]["content"] = f"[ID: {batch_id}]\n\n" + history_with_id[-1]["content"] + f"\n\n**IMPORTANT: Always start your response with [ID: {batch_id}]**"
                
                # Store for batch processing
                batch_requests.append(history_with_id)
                batch_metadata.append({
                    "call_index": call_index,
                    "batch_id": batch_id,
                    "p_call": p_call,
                    "a_call": a_call,
                    "auto_results": auto_results,
                    "llm_needed": llm_needed,
                    "number_mapping": number_mapping,
                    "applied_tool_desc": applied_tool_desc,
                    "history": history
                })
                
                # Store input data for saving
                if save_results:
                    input_data.append({
                        "call_index": call_index,
                        "predicted_call": p_call,
                        "actual_call": a_call,
                        "auto_results": auto_results,
                        "llm_prompt": llm_prompt,
                        "messages": history 
                    })
    
    # Second pass: execute batch LLM calls
    if batch_requests:
        for batch_start in range(0, len(batch_requests), batch_size):
            batch_end = min(batch_start + batch_size, len(batch_requests))
            current_batch = batch_requests[batch_start:batch_end]
            current_metadata = batch_metadata[batch_start:batch_end]
            # Execute batch of LLM calls concurrently
            tasks = [_call_llm_with_semaphore(request, judge_model, gen_config) for request in current_batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process batch results with ID matching
            for i, (llm_result, metadata) in enumerate(zip(batch_results, current_metadata)):
                call_index = metadata["call_index"]
                batch_id = metadata["batch_id"]
                
                try:
                    # Verify response matches request by checking ID
                    response_id = None
                    if llm_result and not isinstance(llm_result, Exception):
                        content = extract_content_from_llm_result(llm_result)
                        id_match = re.search(r'\[ID:\s*([^\]]+)\]', content)
                        if id_match:
                            response_id = id_match.group(1).strip()
                    
                    if response_id != batch_id:
                        logger.error(f"Warning: ID mismatch for call {call_index}. Expected: {batch_id}, Got: {response_id}. Using position-based matching.")
                        # Use position-based matching as fallback
                    
                    result = process_single_llm_result(llm_result, metadata, call_index)
                    all_results.append(result)
                    
                    # Save LLM response if requested
                    if save_results:
                        llm_responses.append({
                            "call_index": call_index,
                            "llm_response": llm_result,
                            "parsed_judgement": result.get('parsed_judgement', {}),
                            "final_metrics": result
                        })
                    
                except Exception as e:
                    logger.error(f"Error processing batch result {call_index}: {e}")
                    error_result = {
                        'call_index': call_index,
                        'error': str(e),
                        'precision': 0.0, 'recall': 0.0, 'f1': 0.0
                    }
                    all_results.append(error_result)
    
    # Process non-LLM cases (auto-only results)
    for global_idx in range(len(predicted_calls)):
        p_call = predicted_calls[global_idx]
        a_call = actual_calls[global_idx]
        call_index = global_idx
        applied_tool_desc = {function_name: tool_descriptions[function_name] for function_name in p_call['function_name'] if function_name in tool_descriptions}

        if not is_rejection_case(a_call, 'label') and not is_rejection_case(p_call, 'prediction') and not is_failed_case(p_call):
            analysis = analyze_argument_matches(p_call, a_call, applied_tool_desc)
            auto_results = analysis['auto_results']
            llm_needed = analysis['llm_needed']
            
            # Skip if already processed in batch
            if not llm_needed or skip_llm_eval:
                final_result = combine_analysis_with_llm_results(analysis, {}, {})
                result = {
                    'call_index': call_index,
                    'precision': final_result['metrics']['precision'],
                    'recall': final_result['metrics']['recall'],
                    'f1': final_result['metrics']['f1']
                }
                all_results.append(result)

    # Save all LLM data if requested
    if save_results and output_dir and file_identifier:
        save_llm_results(input_data, llm_responses, output_dir, file_identifier)
    del(input_data)
    del(llm_responses)

    return all_results

def calculate_function_name_score(predicted_calls: List[Dict], actual_calls: List[Dict], total_label_function_call_count: int) -> Dict[str, Any]:
    """Calculate function name accuracy score by comparing predicted and actual function calls.
    This function evaluates the accuracy of function name predictions by computing F1 scores
    for valid function calls and tracking rejection cases. The input lists contain only 
    function calls that have matching function names between predictions and labels to 
    ensure valid comparisons.
    Args:
        predicted_calls (List[Dict]): List of predicted function call dictionaries.
        actual_calls (List[Dict]): List of actual function call dictionaries.
        total_label_function_call_count (int): Total number of function calls in the label for reference."""
    total_f1 = 0.0
    valid_function_calls = 0
    total_cnt = 0
    for idx in range(len(predicted_calls)):
        p_call = predicted_calls[idx]
        a_call = actual_calls[idx]
        
        # Only calculate F1 for valid function calls (not rejection or failed cases)
        should_reject = is_rejection_case(a_call, "label")
        predicted_reject = is_rejection_case(p_call, "prediction")
        is_failed = is_failed_case(p_call)
        
        if not should_reject and not predicted_reject and not is_failed:
            p_call_function_name = [p_call["function_name"]] if isinstance(p_call["function_name"], str) else p_call["function_name"]
            a_call_function_name = [a_call["function_name"]] if isinstance(a_call["function_name"], str) else a_call["function_name"]
            temp_scores = calculate_f1_score(p_call_function_name, a_call_function_name)
            total_f1 += temp_scores["f1-score"]
            valid_function_calls += 1
            total_cnt += 1

    # Calculate average F1 score only counting succeeded function calls
    avg_f1_score = total_f1 / total_cnt if total_cnt > 0 else 0.0
    
    return {
        "f1-score": round(avg_f1_score, 4),
        "details": {
            "valid_function_calls": valid_function_calls,
            "total_label_function_call_count": total_label_function_call_count
        }
    }
def calculate_key_score(predicted_calls: List[Dict], actual_calls: List[Dict]) -> Dict[str, float]:
    """
    Calculate key score for matched tool call pairs except rejection/failed cases.
    """
    total_f1 = 0.0    
    actual_function_calls = 0
    detailed_key_score = []
    
    for idx in range(len(predicted_calls)):
        p_call = predicted_calls[idx]
        a_call = actual_calls[idx]

        # Check if this should be a rejection case
        should_reject = is_rejection_case(a_call, "label")
        predicted_reject = is_rejection_case(p_call, "prediction")

        # Only calculate for non-rejected cases
        if not should_reject and not predicted_reject:
            p_keys = parse_arguments_to_keys(p_call["arguments"])
            a_keys = parse_arguments_to_keys(a_call["arguments"])
            scores = calculate_f1_score(p_keys, a_keys)
            total_f1 += scores["f1-score"]
            actual_function_calls += 1
            detailed_key_score.append(scores["f1-score"])
        else:
            # rejection case에도 점수를 추가 (일관성을 위해)
            detailed_key_score.append(0.0)
    
    average_f1 = total_f1 / actual_function_calls if actual_function_calls > 0 else 0.0
    return {
        "avg_f1": round(average_f1, 4),
        "detailed_key_score": detailed_key_score
    }


async def calculate_argument_value_scores(predicted_calls: List[Dict],
                                         actual_calls: List[Dict],
                                         histories: List[Dict],
                                         eval_config: Dict,
                                         system_info: Dict,
                                         tool_descriptions: Dict,
                                         save_llm_results: bool, 
                                         output_dir: str, 
                                         file_identifier: str, 
                                         trial: int, 
                                         key_score_result: Dict[str, float],
                                         judge_model: ModelFactory = None,
                                         ):
    """
    Calculate value scores for predicted and actual calls using LLM evaluation.
    """
    detailed_key_score = key_score_result["detailed_key_score"]
    detailed_value_score = []
    
    # 모든 케이스에 대해 초기 스코어 리스트 생성
    for idx in range(len(predicted_calls)):
        detailed_value_score.append(0.0)
    
    try:
        # Filter out rejection cases for LLM evaluation
        llm_eval_pairs = []
        llm_eval_indices = []  # LLM 평가할 인덱스 추적
        
        for idx in range(len(predicted_calls)):
            p_call = predicted_calls[idx]
            a_call = actual_calls[idx]
            history = histories[idx] 
            should_reject = is_rejection_case(a_call, "label")
            predicted_reject = is_rejection_case(p_call, "prediction")
            is_predicted_failed = is_failed_case(p_call)
            
            if not should_reject and not predicted_reject:
                if not is_predicted_failed:
                    # Both are valid function calls - evaluate with LLM
                    llm_eval_pairs.append((p_call, a_call, history))
                    llm_eval_indices.append(idx)
                # else: failed case는 이미 0.0으로 초기화됨
            # else: rejection case는 이미 0.0으로 초기화됨

        # Evaluate valid function call pairs with LLM
        llm_success = False
        if llm_eval_pairs:
            try:
                llm_predicted = [pair[0] for pair in llm_eval_pairs]
                llm_actual = [pair[1] for pair in llm_eval_pairs]
                llm_history = [pair[2] for pair in llm_eval_pairs]

                # Create file identifier for this trial
                trial_file_identifier = f"{file_identifier}_trial_{trial}" if file_identifier else f"trial_{trial}"
                argument_eval = await call_evaluation_llm(
                    llm_predicted, 
                    llm_actual, 
                    eval_config,
                    judge_model, 
                    system_info=system_info,
                    llm_history=llm_history,
                    tool_descriptions=tool_descriptions,
                    save_results=save_llm_results,
                    output_dir=output_dir,
                    file_identifier=trial_file_identifier
                )
                
                if argument_eval:
                    # LLM 결과를 해당 인덱스에 할당
                    for i, result in enumerate(argument_eval):
                        if i < len(llm_eval_indices):
                            idx = llm_eval_indices[i]
                            detailed_value_score[idx] = result.get('f1', 0.0)
                    llm_success = True
                    
            except Exception as llm_error:
                logger.warning(f"LLM evaluation failed, using fallback: {llm_error}")
                logger.debug(f"Detailed traceback: {traceback.format_exc()}")
                llm_success = False

        # LLM이 실패한 경우 fallback 처리
        if not llm_success and llm_eval_pairs:
            for i, (p_call, a_call, _) in enumerate(llm_eval_pairs):
                if i < len(llm_eval_indices):
                    idx = llm_eval_indices[i]
                    p_keys = parse_arguments_to_keys(p_call["arguments"])
                    a_keys = parse_arguments_to_keys(a_call["arguments"])
                    scores = calculate_f1_score(p_keys, a_keys)
                    detailed_value_score[idx] = scores["f1-score"]


        # Calculate average value score (except rejection/failed cases)
        valid_value_scores = [score for score in detailed_value_score if score > 0]
        avg_value_score = sum(valid_value_scores) / len(valid_value_scores) if valid_value_scores else 0.0
        
        perfect_fc_call_result = 0
        min_length = min(len(detailed_key_score), len(detailed_value_score))

        for i in range(min_length):
            if (detailed_key_score[i] == 1.0 and 
                detailed_value_score[i] == 1.0 and
                not is_rejection_case(actual_calls[i], "label")):
                perfect_fc_call_result += 1

        return avg_value_score, perfect_fc_call_result, llm_success
            
    except Exception as e:
        logger.error(f"Error in argument evaluation for trial {trial}: {e}")
        logger.error(traceback.format_exc())
        
        # 완전한 fallback: key-based scoring만 사용
        fallback_value_scores = [0.0] * len(predicted_calls)
        function_call_count = 0
        total_fallback_score = 0.0
        
        for idx in range(len(predicted_calls)):
            p_call = predicted_calls[idx]
            a_call = actual_calls[idx]
            should_reject = is_rejection_case(a_call, "label")
            predicted_reject = is_rejection_case(p_call, "prediction")
            is_predicted_failed = is_failed_case(p_call)
            
            if not should_reject and not predicted_reject and not is_predicted_failed:
                # Both are valid function calls
                p_keys = parse_arguments_to_keys(p_call["arguments"])
                a_keys = parse_arguments_to_keys(a_call["arguments"])
                scores = calculate_f1_score(p_keys, a_keys)
                fallback_value_scores[idx] = scores["f1-score"]
                total_fallback_score += scores["f1-score"]
                function_call_count += 1
        
        # Count successful calls
        min_length = min(len(detailed_key_score), len(fallback_value_scores))
        for i in range(min_length):
            if detailed_key_score[i] == 1.0 and fallback_value_scores[i] == 1.0:
                all_fc_successful_calls += 1
        
        avg_fallback_score = total_fallback_score / function_call_count if function_call_count > 0 else 0.0
        return avg_fallback_score, all_fc_successful_calls, False


async def evaluate_sub_agent_history_f1(data: Dict,
                                        eval_config: Dict,
                                        tool_descriptions: Dict,
                                        judge_model: ModelFactory = None,
                                        save_llm_results: bool = False,
                                        output_dir: str = None,
                                        file_identifier: str = None,) -> Dict:
    """
    Main evaluation function: compares sub_agent_history and label to calculate F1 score.
    """

    num_runs = len(data["history"])
    total_key_scores = 0.0
    total_function_name_scores = 0.0
    total_rejection_cases = 0.0
    total_function_call_cases = 0.0
    total_value_scores_without_rejection = 0.0
    total_all_fc_successful_calls = 0.0
    total_true_positive_reject = 0.0
    total_true_negative_reject = 0.0
    total_false_positive_reject = 0.0
    total_false_negative_reject = 0.0
    total_true_positive_fc = 0  # FC True Positive 누적
    total_false_positive_fc = 0  # FC False Positive 누적
    total_false_negative_fc = 0  # FC False Negative 누적
    total_true_negative_fc = 0  # FC True Negative 누적
    total_rejection_type_mismatch = 0  # Rejection 타입 불일치 누적
    total_failed_generation = 0  # 예측 생성 실패 누적

    total_perfect_successful_calls = 0.0
    llm_failed_trials = []
    for trial in data["history"]:
        data_run = data["history"][trial]
        sub_agent_history = data_run["sub_agent_history"]
        label_history = data_run["label"]
        system_info = label_history["1"]["content"] if label_history["1"]["role"] == "system" else ""

        # Extract tool calls except tool call is different in label and prediction
        (predicted_calls, actual_calls, step_list,
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
            rejection_type_mismatch,
            failed_generation) = \
                            extract_both_tool_calls(sub_agent_history, label_history)

        histories = [create_history(data_run, step) for step in step_list]
        assert len(predicted_calls) == len(actual_calls), f"Mismatch in number of tool calls for trial {trial}"
        # Tool calls 매칭
        # matched_pairs = match_tool_calls_by_similarity(predicted_calls, actual_calls)    
        key_score_result = calculate_key_score(predicted_calls, actual_calls)

        # Calculate function name score using existing function
        function_name_result = calculate_function_name_score(predicted_calls, actual_calls, total_label_fc_cases)
        
        # Accumulate statistics
        total_key_scores += key_score_result["avg_f1"]

        total_function_name_scores += function_name_result["f1-score"] 
        total_rejection_cases += total_label_reject_cases
        total_true_positive_reject += true_positive_reject
        total_true_negative_reject += true_negative_reject
        total_false_positive_reject += false_positive_reject
        total_false_negative_reject += false_negative_reject
        total_true_positive_fc += true_positive_fc
        total_false_positive_fc += false_positive_fc
        total_false_negative_fc += false_negative_fc
        total_true_negative_fc += true_negative_fc
        total_rejection_type_mismatch += rejection_type_mismatch
        total_failed_generation += failed_generation

        total_function_call_cases += total_label_fc_cases

        value_score, perfect_pred_fc_calls, llm_eval_success = await calculate_argument_value_scores(
            predicted_calls, actual_calls, histories, eval_config, 
            system_info, tool_descriptions, save_llm_results, 
            output_dir, file_identifier, trial, key_score_result, judge_model
        )
        total_perfect_successful_calls += perfect_pred_fc_calls
        total_value_scores_without_rejection += value_score
        total_all_fc_successful_calls += true_negative_reject
        if not llm_eval_success: llm_failed_trials.append(trial)

    return {
        "total_perfect_function_calls": total_perfect_successful_calls,
        "total_fc_successful_calls": total_all_fc_successful_calls,
        "key_score": {"avg_f1": round(total_key_scores / num_runs, 4)},
        "value_score_not_count_rejection": {"avg_f1": round(total_value_scores_without_rejection / num_runs, 4)},
        "function_name_score": {"f1-score": round(total_function_name_scores / num_runs, 4)},
        "total_rejection_cases": total_rejection_cases,
        "total_true_positive_reject": total_true_positive_reject,
        "total_true_negative_reject": total_true_negative_reject,
        "total_false_positive_reject": total_false_positive_reject,
        "total_false_negative_reject": total_false_negative_reject,
        "total_true_positive_fc": total_true_positive_fc,  # FC True Positive 총계
        "total_false_positive_fc": total_false_positive_fc,  # FC False Positive 총계
        "total_false_negative_fc": total_false_negative_fc,  # FC False Negative 총계
        "total_true_negative_fc": total_true_negative_fc,  # FC True Negative 총계
        "total_rejection_type_mismatch": total_rejection_type_mismatch,  # Rejection 타입 불일치 총계
        "total_failed_generation": total_failed_generation,  # 예측 생성 실패 총계
        "llm_failed_trials": llm_failed_trials,
        "total_function_call_cases": total_function_call_cases
    }

if __name__ == "__main__":
    import asyncio
    
    async def main():
        file_path = 'data/results/step_wise_evaluation_EN/claude-sonnet-4/16_out.json'
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        eval_prompt_path = "config/base_config/eval_config.yaml" 
        with open(eval_prompt_path, "r", encoding="utf-8") as file:
            eval_config = yaml.safe_load(file)
        
        
        tool_descriptions = {}
        agent_card_pathes = glob.glob("data/EN/multiagent_cards/*.json")
        for path in agent_card_pathes:
            with open(path, 'r', encoding='utf-8') as f:
                agent_card = json.load(f)
            tools = agent_card['tools']
            for tool in tools:
                tool_descriptions[tool['name']] = tool['parameters']
        print(tool_descriptions)
        results = await evaluate_sub_agent_history_f1(data, eval_config, tool_descriptions)
        print(json.dumps(results, indent=2, ensure_ascii=False))
    asyncio.run(main())