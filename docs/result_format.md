# Result Format Documentation

This document describes the structure and meaning of evaluation results produced by OrchestrationBench.

## Overall Structure

```json
{
  "directory": "${output-dir}/${model_name}",
  "overall_statistics": { ... },
  "detailed_results": [ ... ]
}
```

## Field Descriptions

### Key Metrics (Top-Level Summary)

The evaluation provides **4 primary metrics** for quick assessment:

#### 1. Average
- **Formula**: `(Call Rejection Classification Accuracy + FC + Plan) / 3`
- **Description**: Overall performance score combining all three evaluation dimensions.

#### 2. Call Rejection Classification Accuracy
- **Formula**: `(reject_f1 + fc_f1) / 2`
- **Description**: Measures the model's ability to correctly decide between:
  - **Rejection**: When the model should NOT make a function call (e.g., missing required info, constraint violations)
  - **Function Call**: When the model should proceed with tool execution

**Why this metric matters:**
- In real-world scenarios, LLMs must know WHEN to call tools, not just HOW to call them
- A model that always calls tools (ignoring constraints) or never calls tools (being overly cautious) will score poorly
- This metric balances both rejection accuracy (reject_f1) and function call accuracy (fc_f1)

**Rejection cases include:**
- `AWAITING_USER_INPUT`: Model should ask for more information instead of calling a tool
- `TOOL_CONSTRAINT_VIOLATION`: Model should recognize that constraints prevent tool execution (e.g., booking time in the past)

**Example calculation:**
```
reject_f1 = 0.0667  (model correctly identifies 6.67% of rejection cases)
fc_f1 = 0.8382     (model correctly makes function calls 83.82% of the time)
Call Rejection Classification Accuracy = (0.0667 + 0.8382) / 2 = 0.4524
```

#### 3. FC (Function Call)
- **Formula**: `(function_name_f1 + arguments_key_f1 + arguments_value_f1) / 3`
- **Description**: Average F1 score measuring function call quality.
  - `function_name_f1`: F1 score for correct tool/function name selection
  - `arguments_key_f1`: F1 score for correct parameter key matching
  - `arguments_value_f1`: F1 score for correct parameter value matching

#### 4. Plan (Workflow)
- **Formula**: Same as `workflow_evaluation_with_failure`
- **Description**: Quality of multi-step workflow planning using DAG-based evaluation, including penalty for failed workflow generations.

---

### Score Details

#### Workflow Scores
| Field | Description |
|-------|-------------|
| `workflow_evaluation_with_failure` | Overall workflow score including failures (penalizes failed generations) |
| `workflow_evaluation_without_failure` | Overall workflow score excluding workflow generation failures |
| `average_structural_GED` | Workflow structure similarity using Graph Edit Distance (1-GED, higher is better) |
| `average_component_GED` | Workflow component similarity using Graph Edit Distance (1-GED, higher is better) |

#### Function Call Scores
| Field | Description |
|-------|-------------|
| `function_name_accuracy` | F1 score for correct function/tool name selection |
| `arguments_key_score` | F1 score for correct parameter key matching |
| `arguments_value_score_not_count_rejection` | F1 score for correct parameter value matching (excludes rejection cases) |
| `overall_function_call_failed_rate` | `(total_function_call_cases - total_fc_successful_calls) / total_function_call_cases` |
| `total_fc_successful_calls` | Number of successful function calls |
| `total_function_call_cases` | Total number of function call cases in the evaluation |

#### Call/Rejection Scores

##### Rejection Decision Performance
Measures how well the model identifies cases where it should NOT make a function call.

| Field | Formula | Description |
|-------|---------|-------------|
| `precision` | `TP_reject / (TP_reject + FP_reject)` | When model predicts rejection, how often is it correct? |
| `recall` | `TP_reject / (TP_reject + FN_reject)` | Of all actual rejection cases, how many did the model catch? |
| `f1` | `2 * precision * recall / (precision + recall)` | Harmonic mean of precision and recall |
| `accuracy` | `(TP_reject + TN_reject) / (TP + TN + FP + FN)` | Overall rejection decision accuracy |

##### FC Decision Performance
Measures how well the model correctly executes function calls.

| Field | Formula | Description |
|-------|---------|-------------|
| `precision` | `FC_TP / (FC_TP + FC_FP)` | When model makes a function call, how often should it have? |
| `recall` | `FC_TP / (FC_TP + FC_FN)` | Of all cases requiring function calls, how many did the model make? |
| `f1` | `2 * precision * recall / (precision + recall)` | Harmonic mean of precision and recall |
| `accuracy` | `(FC_TP + FC_TN) / (FC_TP + FC_FP + FC_FN + FC_TN)` | Overall FC decision accuracy |

##### Error Patterns
| Field | Description |
|-------|-------------|
| `overaction_rate` | Rate of unnecessary actions - model called a tool when it shouldn't have (FN_reject / total_errors) |
| `underaction_rate` | Rate of missed necessary actions - model rejected when it should have called (FP_reject / total_errors) |
| `type_mismatch_rate` | Rate of rejection type mismatches |

##### Rejection Type Accuracy
| Field | Formula | Description |
|-------|---------|-------------|
| `rejection_type_accuracy` | `(TP_reject - type_mismatch) / TP_reject` | When model correctly rejects, how often does it predict the correct rejection type? |
| `total_rejection_type_mismatch` | - | Count of cases where model correctly rejected but with wrong type (e.g., predicted `AWAITING_USER_INPUT` but actual was `TOOL_CONSTRAINT_VIOLATION`) |
| `total_failed_generation` | - | Count of cases where model failed to generate a valid prediction (neither FC nor Reject) |

---

### Raw Confusion Matrix

The confusion matrix tracks predictions vs actual labels for both rejection and function call perspectives.

#### Rejection Perspective
| Field | Description |
|-------|-------------|
| `total_true_positive_reject` | Model correctly predicted rejection (actual: reject, predicted: reject) |
| `total_true_negative_reject` | Model correctly predicted function call (actual: FC, predicted: FC) |
| `total_false_positive_reject` | Model incorrectly predicted rejection (actual: FC, predicted: reject) |
| `total_false_negative_reject` | Model incorrectly predicted function call (actual: reject, predicted: FC) |

#### Function Call Perspective
| Field | Description |
|-------|-------------|
| `total_true_positive_fc` | Actual FC, Predicted FC (correct function call) |
| `total_false_positive_fc` | Actual reject, Predicted FC (should have rejected) |
| `total_false_negative_fc` | Actual FC, Predicted reject (should have called) |
| `total_true_negative_fc` | Actual reject, Predicted reject (correct rejection) |

---

### Visual Representation

The model can make 3 types of predictions: **FC** (Function Call), **Reject**, or **FAILED** (generation failure).

```
                              PREDICTED
                    FC           Reject          FAILED
              ┌───────────┬─────────────────┬───────────┐
         FC   │  TP_fc    │     FN_fc       │   FN_fc   │
ACTUAL        │  TN_rej   │     FP_rej      │  (failed) │
              ├───────────┼─────────────────┼───────────┤
       Reject │  FP_fc    │ TP_rej(+type_mm)│  (failed) │
              │  FN_rej   │     TN_fc       │   TN_fc   │
              └───────────┴─────────────────┴───────────┘

where:
  type_mm = rejection_type_mismatch (predicted reject but wrong type)
  failed  = failed_generation (model failed to produce valid output)

Key relationships:
  - TP_reject includes type_mismatch (reject decision was correct, only type was wrong)
  - TN_fc = TP_reject (including type_mismatch cases)
  - FAILED counts separately, not included in FP_reject
```

**Prediction Categories:**

| Predicted | Actual FC | Actual Reject |
|-----------|-----------|---------------|
| FC | TP_fc, TN_reject | FP_fc, FN_reject |
| Reject (correct type) | FP_reject, FN_fc | TP_reject, TN_fc |
| Reject (wrong type) | FP_reject, FN_fc | TP_reject + type_mismatch, TN_fc |
| FAILED | failed_generation, FN_fc | failed_generation, TN_fc |

**Note**:
- `type_mismatch` is counted as `TP_reject` because the model correctly decided to reject (just with wrong type)
- `rejection_type_accuracy` measures how often the rejection type was correct among successful rejections
- `FAILED` means model failed to generate valid output, tracked separately from rejection predictions
