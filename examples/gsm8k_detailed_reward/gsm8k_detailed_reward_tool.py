"""GSM8K tool with tiered rewards and detailed feedback.

Differences from Gsm8kTool:
- Partial credit (0.5) for numerically close answers (within 5%)
- Descriptive feedback text so the model knows *why* it scored what it did
- Final reward = best reward across all attempts
"""

import logging
import os
from typing import Any, Optional
from uuid import uuid4

from verl.utils.reward_score.gsm8k import extract_solution
from verl.utils.rollout_trace import rollout_trace_op

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class Gsm8kDetailedRewardTool(BaseTool):
    """GSM8K tool that returns tiered rewards with detailed feedback."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        if ground_truth is None:
            ground_truth = kwargs.get("create_kwargs", {}).get("ground_truth", None)
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        answer = parameters.get("answer", "")
        if not isinstance(answer, str):
            answer = str(answer)

        if answer.startswith("#### "):
            self._instance_dict[instance_id]["response"] = answer
        else:
            self._instance_dict[instance_id]["response"] = "#### " + answer

        reward = await self.calc_reward(instance_id)
        prev_reward = self._instance_dict[instance_id]["reward"]
        # penalty for non-improved answer submission (>= so repeating a correct answer isn't penalized)
        tool_reward = 0.0 if reward >= prev_reward else -0.05
        # keep best reward
        self._instance_dict[instance_id]["reward"] = max(prev_reward, reward)

        ground_truth = self._instance_dict[instance_id]["ground_truth"]
        parsed = extract_solution(self._instance_dict[instance_id]["response"], method="flexible")

        if reward >= 1.0:
            feedback = f"Correct! Your answer {answer} matches the expected answer {ground_truth}. Reward: {reward}"
        elif reward >= 0.5:
            feedback = (
                f"Close but not exact. Your answer {answer} (parsed as {parsed}) is within 5% of the expected "
                f"answer. Check your arithmetic for rounding or off-by-one errors. Reward: {reward}"
            )
        elif reward > 0.0:
            feedback = (
                f"Incorrect. Your answer {answer} (parsed as {parsed}) does not match the expected answer. "
                f"Please re-read the problem and try a different approach. Reward: {reward}"
            )
        else:
            feedback = (
                f"Could not parse a numeric answer from '{answer}'. "
                f"Please provide a plain number (e.g. 42 or -3.5). Reward: {reward}"
            )

        return ToolResponse(text=feedback), tool_reward, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        response = self._instance_dict[instance_id]["response"]
        ground_truth = self._instance_dict[instance_id]["ground_truth"]

        # Uses "flexible" extraction (finds the last number anywhere) because
        # execute() prepends "#### " itself — the model's raw answer won't have it.
        # The final episode reward uses a separate compute_score with "strict" extraction.
        parsed = extract_solution(response, method="flexible")
        if parsed is None:
            return 0.0

        # Normalize commas for comparison — extract_solution("flexible") may
        # return "1,200" while ground_truth is always comma-stripped by the preprocessor.
        parsed_normalized = parsed.replace(",", "")

        # Exact match
        if parsed_normalized == ground_truth:
            return 1.0

        # Numeric closeness check
        try:
            pred_val = float(parsed_normalized)
            gt_val = float(ground_truth.replace(",", ""))
            if gt_val != 0 and abs(pred_val - gt_val) / abs(gt_val) <= 0.05:
                return 0.5
        except ValueError:
            pass

        # Parsed but wrong
        return 0.1

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
