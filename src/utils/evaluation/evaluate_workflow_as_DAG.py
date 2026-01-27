import json
import re
import yaml
from loguru import logger
from typing import Dict, List, Any, Tuple, Set, Optional
import traceback
from collections import defaultdict
import difflib
import networkx as nx
import math


def _unit_node_subst_cost(a1, a2, keys=None):
    """
    Returns 0 if all attributes in keys are identical, 1 if any differ
    (all weights are fixed to 1)
    """
    if not keys:
        return 0
    return 0 if all(a1.get(k) == a2.get(k) for k in keys) else 1

def _weighted_node_subst_cost(a1: Dict[str, Any], a2: Dict[str, Any], agent_weight: float, status_weight: float) -> float:
    """
    Agent-state node substitution cost (with weight support)
    
    Args:
        a1, a2: Node attribute dictionaries (name: agent_id, status: status)
        agent_weight: Cost weight when agent_id changes
        status_weight: Cost weight when status changes
    
    Returns:
        float: Substitution cost (0~1)
    """
    agent1 = a1.get('name', '')
    agent2 = a2.get('name', '')
    status1 = a1.get('status', '')
    status2 = a2.get('status', '')
    
    # Zero cost if completely identical
    if agent1 == agent2 and status1 == status2:
        return 0.0
    
    # Cost 1 if completely different
    if agent1 != agent2 and status1 != status2:
        return 1.0
    
    # Same agent_id but different status
    if agent1 == agent2 and status1 != status2:
        return status_weight
    
    # Same status but different agent_id
    if agent1 != agent2 and status1 == status2:
        return agent_weight
    
    return 1.0  # Default value

def _unit_edge_subst_cost(e1: Dict[str, Any], e2: Dict[str, Any], keys=None) -> float:
    # Returns 0 for structure only, specify keys to check specific edge attributes
    if not keys:
        return 0
    return 0 if all(e1.get(k) == e2.get(k) for k in keys) else 1

def _max_edit_cost(G1: nx.Graph, G2: nx.Graph, n_del=1, n_ins=1, e_del=1, e_ins=1):
    return len(G1.nodes())*n_del + len(G1.edges())*e_del + \
           len(G2.nodes())*n_ins + len(G2.edges())*e_ins

def _ged_similarity(G1: nx.Graph, G2: nx.Graph, node_keys=None, edge_keys=None, upper_bound=None, 
                   use_weighted_cost=False, agent_weight=0.5, status_weight=0.5):
    """
    GED-based similarity calculation
    - Compatible handling for node/edge substitution cost callbacks whether they receive 'attribute dict' or 'node/edge identifier'
    - Conservatively replace with max_cost when edit_cost is None/inf/nan (similarity 0)
    - Set upper_bound default to max_cost to prevent excessive search expansion
    """
    def _node_attrs(x, G):
        # Return as-is if dict, lookup attributes from G if identifier
        return x if isinstance(x, dict) else G.nodes[x]

    def _edge_attrs(x, G):
        # Return as-is if dict, lookup attributes from G if identifier (e.g., (u,v))
        return x if isinstance(x, dict) else G.edges[x]

    def node_subst(u, v):
        a1 = _node_attrs(u, G1)
        a2 = _node_attrs(v, G2)
        if use_weighted_cost and node_keys and 'name' in node_keys and 'status' in node_keys:
            return _weighted_node_subst_cost(a1, a2, agent_weight, status_weight)
        return _unit_node_subst_cost(a1, a2, node_keys)

    def edge_subst(e1, e2):
        d1 = _edge_attrs(e1, G1)
        d2 = _edge_attrs(e2, G2)
        return _unit_edge_subst_cost(d1, d2, edge_keys)

    max_cost = _max_edit_cost(G1, G2)

    # Use max_cost as upper_bound if not specified (upper limit for full deletion + full insertion)
    ub = max_cost if upper_bound is None else upper_bound

    edit_cost = nx.graph_edit_distance(
        G1, G2,
        node_subst_cost=node_subst,
        node_del_cost=lambda _: 1,
        node_ins_cost=lambda _: 1,
        edge_subst_cost=edge_subst,
        edge_del_cost=lambda _: 1,
        edge_ins_cost=lambda _: 1,
        upper_bound=ub
    )

    # Defense against failure/abort/numerical anomalies
    if edit_cost is None or not math.isfinite(edit_cost):
        edit_cost = float(max_cost)

    sim = 1 - (edit_cost / max_cost) if max_cost > 0 else 1.0
    return float(edit_cost), float(max_cost), float(sim)

# ---------- 1) Workflow Structure (DAG) ----------
def build_workflow_graph(workflow_data: Dict[str, Any]) -> nx.DiGraph:
    """
    Nodes: workflow names
    Edges: depends_on (dep -> cur)
    Structure only (no node/edge attributes)
    """
    G = nx.DiGraph()
    for wf, attrs in workflow_data.items():
        G.add_node(wf)
        for dep in attrs.get('depends_on', []) or []:
            G.add_edge(dep, wf)
    return G

def remove_key_except_workflow(workflow_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Removes all keys except those specified in keys_to_keep from the workflow data.
    """
    return {k: v for k, v in workflow_data.items() if "workflow" in k}

def evaluate_workflow_structure(pred_workflows: Dict[str, Any], gold_workflows: Dict[str, Any], upper_bound=None) -> Dict[str, float]:

    pred_workflows = remove_key_except_workflow(pred_workflows)
    gold_workflows = remove_key_except_workflow(gold_workflows)
    Gp = build_workflow_graph(pred_workflows)
    Gl = build_workflow_graph(gold_workflows)
    # Structure only comparison → substitution cost always 0, insertion/deletion 1
    return _ged_similarity(Gp, Gl, node_keys=None, edge_keys=None, upper_bound=upper_bound)

# ---------- 2) Agent-State Hierarchy ----------
def _first_agent_state(steps: List[Dict[str, Any]]) -> Optional[Tuple[str, str]]:
    for s in steps or []:
        a = s.get('agent_id') or s.get('name')  # Handle both 'agent_id' and 'name' keys
        st = s.get('status', 'UNK')
        if a:
            return a, st
    return None

def _last_agent_state(steps: List[Dict[str, Any]]) -> Optional[Tuple[str, str]]:
    last = None
    for s in steps or []:
        a = s.get('agent_id') or s.get('name')  # Handle both 'agent_id' and 'name' keys
        st = s.get('status', 'UNK')
        if a:
            last = (a, st)
    return last

def build_agent_state_graph(workflow_data: Dict[str, Any]) -> nx.DiGraph:
    """
    Nodes: (agent_id, status) pairs -> attrs: {'name': agent_id, 'status': status, 'label': f'{agent_id}|{status}'}
    Edges:
      (1) Sequential steps within same workflow: (a_i, st_i) -> (a_{i+1}, st_{i+1})
      (2) depends_on: dep last (a_dep, st_dep) -> current first (a_cur, st_cur)
    ※ Same (agent,status) appearing multiple times aggregated as single node (DiGraph)
    """
    G = nx.DiGraph()
    
    # Helper function to get agent_id from step (handles both 'agent_id' and 'name' keys)
    def get_agent_id(step):
        return step.get('agent_id') or step.get('name')
    
    # Extract nodes first
    for wf, attrs in workflow_data.items():
        for s in attrs.get('steps', []) or []:
            a = get_agent_id(s)
            st = s.get('status', 'UNK')
            if not a:
                continue
            key = (a, st)
            if key not in G:
                G.add_node(key, name=a, status=st, label=f"{a}|{st}")

    # Internal workflow connections
    for wf, attrs in workflow_data.items():
        seq = [(get_agent_id(s), s.get('status', 'UNK'))
               for s in (attrs.get('steps') or []) if get_agent_id(s)]
        for (a1, st1), (a2, st2) in zip(seq, seq[1:]):
            
            if (a1, st1) not in G:
                G.add_node((a1, st1), name=a1, status=st1, label=f"{a1}|{st1}")
            if (a2, st2) not in G:
                G.add_node((a2, st2), name=a2, status=st2, label=f"{a2}|{st2}")
            if (a1, st1) != (a2, st2):
                G.add_edge((a1, st1), (a2, st2))

    # Inter-workflow depends_on connections (handle both 'depends_on' and 'depend_on')
    for wf, attrs in workflow_data.items():
        cur_first = _first_agent_state(attrs.get('steps'))
        if not cur_first:
            continue
        
        # Handle both depends_on (list) and depend_on (single value)
        dependencies = attrs.get('depends_on', []) or []
        if not dependencies:
            depend_on = attrs.get('depend_on')
            if depend_on:
                dependencies = [depend_on] if isinstance(depend_on, str) else depend_on
        
        for dep in dependencies:
            dep_steps = (workflow_data.get(dep, {}) or {}).get('steps', [])
            dep_last = _last_agent_state(dep_steps)
            if dep_last:
                if dep_last not in G:
                    G.add_node(dep_last, name=dep_last[0], status=dep_last[1], label=f"{dep_last[0]}|{dep_last[1]}")
                if cur_first not in G:
                    G.add_node(cur_first, name=cur_first[0], status=cur_first[1], label=f"{cur_first[0]}|{cur_first[1]}")
                G.add_edge(dep_last, cur_first)

    return G

def evaluate_agent_state_hierarchy(pred_workflows: Dict[str, Any], gold_workflows: Dict[str, Any], upper_bound=None, 
                                  use_weighted_cost=False, agent_weight:float=0.5, status_weight:float=0.5):
    """
    Node substitution cost: 
    - use_weighted_cost=False: 0 if both (agent_id, status) match, 1 otherwise
    - use_weighted_cost=True: status_weight if same agent_id but different status, 1 if completely different
    Edge substitution cost: structure only (no attributes) → 0
    Insertion/deletion: 1
    """
    pred_workflows = remove_key_except_workflow(pred_workflows)
    gold_workflows = remove_key_except_workflow(gold_workflows)
    Gp = build_agent_state_graph(pred_workflows)
    Gl = build_agent_state_graph(gold_workflows)
    
    return _ged_similarity(Gp, Gl, node_keys=['name', 'status'], edge_keys=None, 
                          upper_bound=upper_bound, use_weighted_cost=use_weighted_cost,
                          agent_weight=agent_weight, status_weight=status_weight)

# ---------- 3) Comprehensive Evaluation Function ----------
def evaluate_workflow_similarity(pred_workflows: Dict[str, Any], gold_workflows: Dict[str, Any], 
                                workflow_weight: float = 0.5, agent_weight: float = 0.5, 
                                upper_bound: Optional[float] = None,
                                use_weighted_agent_cost: bool = False, 
                                agent_change_weight: float = 0.5, 
                                status_change_weight: float = 0.5):
    """
    Comprehensive workflow similarity evaluation
    
    Args:
        pred_workflows: Predicted workflow data
        gold_workflows: Ground truth workflow data
        workflow_weight: Workflow structure weight
        agent_weight: Agent-state hierarchy weight
        upper_bound: GED calculation upper bound (for performance optimization)
        use_weighted_agent_cost: Whether to use weighted agent-state cost
        agent_change_weight: Cost for agent_id changes (when use_weighted_agent_cost=True)
        status_change_weight: Cost for status changes (when use_weighted_agent_cost=True)
    
    Returns:
        dict: Evaluation results
    """
    assert abs(workflow_weight + agent_weight - 1.0) < 1e-6, "Weights must sum to 1.0"
    
    # 1. Workflow structure evaluation
    w_edit, w_max, w_sim = evaluate_workflow_structure(pred_workflows, gold_workflows, upper_bound)
    
    # 2. Agent-state hierarchy evaluation  
    a_edit, a_max, a_sim = evaluate_agent_state_hierarchy(
        pred_workflows, gold_workflows, upper_bound,
        use_weighted_cost=use_weighted_agent_cost,
        agent_weight=agent_change_weight,
        status_weight=status_change_weight
    )
    
    # 3. Overall score
    overall_similarity = w_sim * workflow_weight + a_sim * agent_weight
    
    return {
        'overall_similarity': overall_similarity,
        'workflow_structure': {
            'edit_cost': w_edit,
            'max_cost': w_max, 
            'similarity': w_sim
        },
        'agent_state_hierarchy': {
            'edit_cost': a_edit,
            'max_cost': a_max,
            'similarity': a_sim
        },
        'weights': {
            'workflow_weight': workflow_weight,
            'agent_weight': agent_weight
        },
        'agent_cost_settings': {
            'use_weighted_cost': use_weighted_agent_cost,
            'agent_change_weight': agent_change_weight,
            'status_change_weight': status_change_weight
        }
    }

# ---------- 4) Detailed Analysis Function ----------
def detailed_analysis(pred_workflows, gold_workflows):
    """Provide detailed analysis results"""
    
    # Workflow graph analysis
    pred_wf_graph = build_workflow_graph(pred_workflows)
    gold_wf_graph = build_workflow_graph(gold_workflows)
    
    # Agent-state graph analysis
    pred_as_graph = build_agent_state_graph(pred_workflows)
    gold_as_graph = build_agent_state_graph(gold_workflows)
    
    analysis = {
        'workflow_analysis': {
            'pred_workflow_count': len(pred_workflows),
            'gold_workflow_count': len(gold_workflows),
            'pred_dependencies': pred_wf_graph.number_of_edges(),
            'gold_dependencies': gold_wf_graph.number_of_edges(),
            'common_workflows': list(set(pred_workflows.keys()) & set(gold_workflows.keys())),
            'missing_workflows': list(set(gold_workflows.keys()) - set(pred_workflows.keys())),
            'extra_workflows': list(set(pred_workflows.keys()) - set(gold_workflows.keys()))
        },
        'agent_state_analysis': {
            'pred_agent_state_nodes': pred_as_graph.number_of_nodes(),
            'gold_agent_state_nodes': gold_as_graph.number_of_nodes(),
            'pred_agent_state_edges': pred_as_graph.number_of_edges(),
            'gold_agent_state_edges': gold_as_graph.number_of_edges(),
            'pred_agent_states': [f"{attrs['name']}|{attrs['status']}" 
                                for node, attrs in pred_as_graph.nodes(data=True)],
            'gold_agent_states': [f"{attrs['name']}|{attrs['status']}" 
                                for node, attrs in gold_as_graph.nodes(data=True)]
        }
    }
    
    return analysis

def extract_workflow_from_content(content: str) -> Dict[str, Any]:
    """
    Extracts a workflow JSON/YAML string from agent content and parses it.
    This handles raw JSON, JSON in markdown, plain YAML, and YAML in markdown.
    """
    if not isinstance(content, str):
        return {} # Not a string, cannot parse

    content = content.split("</think>")[-1]
    
    # First, try to extract JSON from markdown code blocks (```json ... ```)
    json_md_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL|re.MULTILINE)
    if json_md_match:
        try:
            parsed = json.loads(json_md_match.group(1))
            if isinstance(parsed, dict) and any(key.startswith('workflow_') for key in parsed.keys()):
                return parsed   
        except json.JSONDecodeError:
            pass

    # Try to find JSON that starts with { and ends with } (for unescaped JSON in content)
    json_pattern = re.search(r'(\{[^}]*"workflow_[^}]*\}(?:\s*,\s*\{[^}]*"workflow_[^}]*\})*)', content, re.DOTALL)
    if json_pattern:
        try:
            # Try to parse as a single JSON object or fix malformed JSON
            json_str = json_pattern.group(1)
            # If it looks like multiple objects, wrap them in an outer object
            if json_str.count('"workflow_') > 1 and not json_str.strip().startswith('{'):
                json_str = '{' + json_str + '}'
            parsed = json.loads(json_str)
            if isinstance(parsed, dict) and any(key.startswith('workflow_') for key in parsed.keys()):
                return parsed
        except json.JSONDecodeError:
            pass

    # More aggressive JSON extraction - look for workflow patterns and try to reconstruct
    workflow_pattern = re.findall(r'"(workflow_\d+)":\s*\{[^}]*\}', content, re.DOTALL)
    if workflow_pattern:
        # Try to extract the entire JSON structure containing workflows
        start_idx = content.find('"workflow_')
        if start_idx > 0:
            # Find the opening brace before the workflow
            brace_start = content.rfind('{', 0, start_idx)
            if brace_start >= 0:
                # Find matching closing brace
                brace_count = 0
                end_idx = brace_start
                for i, char in enumerate(content[brace_start:], brace_start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                
                try:
                    potential_json = content[brace_start:end_idx]
                    parsed = json.loads(potential_json)
                    if isinstance(parsed, dict) and any(key.startswith('workflow_') for key in parsed.keys()):
                        return parsed
                except json.JSONDecodeError:
                    pass

    # Attempt to parse as direct JSON
    try:
        potential_json = json.loads(content)
        if isinstance(potential_json, dict) and any(key.startswith('workflow_') for key in potential_json.keys()):
            return potential_json
    except json.JSONDecodeError:
        pass

    # Attempt to extract from markdown yaml block
    yaml_md_match = re.search(r"```yaml\s*(workflow_.*?)\s*```", content, re.DOTALL)
    if yaml_md_match:
        try:
            parsed = yaml.safe_load(yaml_md_match.group(1))
            if isinstance(parsed, dict) and any(key.startswith('workflow_') for key in parsed.keys()):
                return parsed
        except yaml.YAMLError:
            pass

    # Enhanced YAML parsing - try to parse the entire content as YAML if it contains workflow_
    if "workflow_" in content:
        try:
            parsed = yaml.safe_load(content)
            if isinstance(parsed, dict) and any(key.startswith('workflow_') for key in parsed.keys()):
                return parsed
        except yaml.YAMLError:
            pass
        
        # Try to fix common YAML formatting issues
        try:
            # Handle cases where the YAML isn't properly indented
            lines = content.strip().split('\n')
            if lines and lines[0].startswith('workflow_'):
                # Add proper indentation for YAML parsing
                formatted_content = ""
                for line in lines:
                    if line.strip().startswith('workflow_'):
                        formatted_content += line + ":\n"
                    elif ':' in line and not line.strip().startswith('-'):
                        formatted_content += "  " + line.strip() + "\n"
                    elif line.strip().startswith('-'):
                        formatted_content += "    " + line.strip() + "\n"
                    else:
                        formatted_content += "      " + line.strip() + "\n"
                
                parsed = yaml.safe_load(formatted_content)
                if isinstance(parsed, dict) and any(key.startswith('workflow_') for key in parsed.keys()):
                    return parsed
        except yaml.YAMLError:
            pass

    # Attempt to parse as plain YAML (legacy support)
    try:
        # Heuristic: If it starts with 'workflow_', it's likely a YAML workflow definition
        if content.strip().startswith("workflow_"): 
            parsed = yaml.safe_load(content)
            if isinstance(parsed, dict) and any(key.startswith('workflow_') for key in parsed.keys()):
                return parsed
    except yaml.YAMLError:
        pass

    return {} # Return empty dict if no valid workflow is found

def _create_zero_score_metrics(step_id: int, agent_change_weight: float, status_change_weight: float) -> Dict[str, Any]:
    """Create zero-score metrics for cases where no prediction exists or prediction is invalid."""
    return {
        "step_id": step_id,
        'overall_similarity': 0.0,
        'workflow_structure': {
            'edit_cost': 0.0,
            'max_cost': 0.0,
            'similarity': 0.0
        },
        'agent_state_hierarchy': {
            'edit_cost': 0.0,
            'max_cost': 0.0,
            'similarity': 0.0
        },
        'weights': {
            'workflow_weight': 0.5, 
            'agent_weight': 0.5
        },
        'agent_cost_settings': {
            'use_weighted_cost': True,
            'agent_change_weight': agent_change_weight,
            'status_change_weight': status_change_weight
        }
    }

def evaluate_run(run_data: Dict[str, Any], agent_change_weight: float, status_change_weight: float) -> Tuple[List[Dict[str, Any]], int]:
    """
    Evaluates a single run by finding all predicted workflow definitions in 
    main_agent_history and comparing them with corresponding ground truth 
    definitions in the label, based on step_id.
    
    Returns a tuple: (list of metrics dicts for each step_id, number of failed workflow generations).
    """
    main_agent_history = run_data.get('main_agent_history', [])
    label = run_data.get('label', {})
    step_metrics_list = []
    failed_workflow_generation = 0
    
    # Map main_agent_history step_id to predicted workflow dict
    predicted_workflows_by_step: Dict[int, Dict[str, Any]] = {}
    ground_truth_workflow_steps: Dict[int, Dict[str, Any]] = {}
    
    # Extract predicted workflows from main_agent_history
    for step_entry in main_agent_history:
        step_id = step_entry.get('step_id')
        if step_id is None:
            continue
        content = step_entry.get('content')
        if content is not None:
            extracted_workflow = extract_workflow_from_content(content)
            # Check if the extracted content actually contains workflow definitions
            if any(key.startswith('workflow_') for key in extracted_workflow.keys()):
                predicted_workflows_by_step[step_id] = extracted_workflow
                logger.debug(f"Found predicted workflow for step {step_id}: {extracted_workflow}")
            else:
                logger.debug(f"No workflow found in step {step_id} content")
                failed_workflow_generation += 1
        else:
            failed_workflow_generation += 1
            continue
    
    # Extract ground truth workflows from label
    for label_key, label_data in label.items():
        try:
            step_id = int(label_key)
            label_content = label_data.get('content', '')
            if isinstance(label_content, str) and 'workflow_' in label_content:
                extracted_gt_workflow = extract_workflow_from_content(label_content)
                if any(key.startswith('workflow_') for key in extracted_gt_workflow.keys()):
                    ground_truth_workflow_steps[step_id] = extracted_gt_workflow
                    logger.debug(f"Found ground truth workflow for step {step_id}: {extracted_gt_workflow}")
        except (ValueError, TypeError):
            # Skip non-numeric keys
            continue
    
    logger.debug(f"Found {len(predicted_workflows_by_step)} predicted workflows and {len(ground_truth_workflow_steps)} ground truth workflows")
    
    # Compare workflows for each ground truth step
    for gt_step_id in sorted(ground_truth_workflow_steps.keys()):
        ground_truth_workflow = ground_truth_workflow_steps[gt_step_id]
        
        if gt_step_id in predicted_workflows_by_step:
            predicted_workflow = predicted_workflows_by_step[gt_step_id]
            if not predicted_workflow or type(predicted_workflow) is not dict:
                # If no prediction was made by the main agent at that exact step_id, assign 0.0 scores.
                step_metrics_list.append(_create_zero_score_metrics(gt_step_id, agent_change_weight, status_change_weight))
                failed_workflow_generation += 1
                continue
            
            logger.debug(f"Comparing workflows for step {gt_step_id}")
            # Calculate structural accuracy for this specific step_id
            results = evaluate_workflow_similarity(
                predicted_workflow, ground_truth_workflow,
                use_weighted_agent_cost=True,
                agent_change_weight=agent_change_weight,     # Cost 1.0 when agent_id changes
                status_change_weight=status_change_weight     # Cost 0.5 when status changes
            )
            combined_metrics = {
                "step_id": gt_step_id,
                **results
            }
            step_metrics_list.append(combined_metrics)
            logger.debug(f"Step {gt_step_id} metrics: {combined_metrics}")
        else:
            # If a ground truth workflow exists for this step_id but no prediction was made
            # by the main agent at that exact step_id, assign 0.0 scores.
            failed_workflow_generation += 1
            step_metrics_list.append(_create_zero_score_metrics(gt_step_id, agent_change_weight, status_change_weight))
            logger.debug(f"No prediction found for ground truth step {gt_step_id}")

    return step_metrics_list, failed_workflow_generation

def evaluate_workflow_multiple_runs(data: Dict[str, Any], agent_change_weight: float, status_change_weight: float) -> Dict[str, float]:
    """
    Evaluates multiple runs (e.g., run #1 to run #5) based on step-by-step workflow comparisons
    and returns average accuracy metrics including both structural and status evaluations.
    """
    # Structural metrics

    total_scores = []
    workflow_GED_scores = []
    agents_GED_scores = []

    history = data.get('history', {})
    failed_workflow_generation = 0
    total_comparison_steps_evaluated = 0

    for i in range(1, len(history) + 1):
        run_key = f"run #{i}"
        try:
            run_step_metrics, failed_workflow_gen_in_single_run = evaluate_run(history[run_key], agent_change_weight, status_change_weight)
            failed_workflow_generation += failed_workflow_gen_in_single_run
            for metrics in run_step_metrics:
                total_scores.append(metrics['overall_similarity'])
                workflow_GED_scores.append(metrics['workflow_structure']['similarity'])
                agents_GED_scores.append(metrics['agent_state_hierarchy']['similarity'])
                total_comparison_steps_evaluated += 1
                
        except Exception as e:
            logger.debug(traceback.format_exc())
            # Log the error but continue with other runs
            logger.debug(f"Warning: Failed to evaluate {run_key}: {e}")
            continue
    
    # Calculate structural averages
    avg_total_scores_with_failure = sum(total_scores) / (total_comparison_steps_evaluated + failed_workflow_generation) if (total_comparison_steps_evaluated + failed_workflow_generation) else 0.0
    avg_total_scores = sum(total_scores) / total_comparison_steps_evaluated if total_comparison_steps_evaluated else 0.0
    avg_workflow_scores = sum(workflow_GED_scores) / total_comparison_steps_evaluated if total_comparison_steps_evaluated else 0.0
    avg_agents_scores = sum(agents_GED_scores) / total_comparison_steps_evaluated if total_comparison_steps_evaluated else 0.0

    return {
        # Structural metrics
        "total_score_with_failure": avg_total_scores_with_failure,
        "total_score_without_failure": avg_total_scores,
        "average_structural_GED": avg_workflow_scores,
        "average_component_GED": avg_agents_scores,
        # Error tracking
        "failed_workflow_generation": failed_workflow_generation,
        # Total comparison count
        "total_comparison_steps_evaluated": total_comparison_steps_evaluated
    }


if __name__ == "__main__":
    # Existing DAG evaluation test
    import glob
    file_paths = glob.glob('data/results/step_wise_evaluation/claude*/134_out.json')
    all_sample_data = []

    for file_path in file_paths:
        print(file_path)
        with open(file_path, 'r', encoding='utf-8') as file:
            sample_data = json.load(file)
            
        average_metrics = evaluate_workflow_multiple_runs(sample_data, 0.8, 0.5)
        all_sample_data.append(average_metrics)
        break
    print("Existing DAG evaluation results:", all_sample_data[0])
    
    