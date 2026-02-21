"""Custom GSM8K reward with tiered scoring for single-turn training.

Differences from the default binary reward (verl.utils.reward_score.gsm8k):
- 1.0  for exact match
- 0.5  for numerically close (within 5%)
- 0.1  for parseable but wrong (format credit)
- 0.0  for unparseable

Usage:
    reward.custom_reward_function.path=examples/gsm8k_custom_reward/gsm8k_custom_reward.py
    reward.custom_reward_function.name=compute_score
"""

from __future__ import annotations

from typing import Any, Optional

from verl.utils.reward_score.gsm8k import extract_solution


def compute_score(
    data_source: Optional[str],
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict[str, Any]] = None,
    **kwargs,
) -> float:
    # Uses "strict" extraction — requires the model to produce "#### <number>" format.
    # This matches the default gsm8k scorer's expectation.
    answer = extract_solution(solution_str, method="strict")
    if answer is None:
        return 0.0

    # Exact match
    if answer == ground_truth:
        return 1.0

    # Numeric closeness check
    try:
        pred_val = float(answer.replace(",", ""))
        gt_val = float(ground_truth.replace(",", ""))
        if gt_val != 0 and abs(pred_val - gt_val) / abs(gt_val) <= 0.05:
            return 0.5
    except ValueError:
        pass

    # Parsed but wrong — format credit
    return 0.1
