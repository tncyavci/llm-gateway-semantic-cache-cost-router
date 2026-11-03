from __future__ import annotations

import re
from typing import Dict

from .models import LengthGateSpec, LengthValidationResult

_WORD_TOKEN_RE = re.compile(r"\S+")

DEFAULT_TASK_LENGTH_SPECS: Dict[str, LengthGateSpec] = {
    "long_form_passage": LengthGateSpec(
        task_type="long_form_passage", min_words=400, max_words=500
    ),
    "executive_brief": LengthGateSpec(
        task_type="executive_brief", min_words=180, max_words=240
    ),
    "clinical_case_summary": LengthGateSpec(
        task_type="clinical_case_summary", min_words=250, max_words=320
    ),
}


def count_words(text: str) -> int:
    return len(_WORD_TOKEN_RE.findall(text.strip()))


class CeilingLengthGate:
    """
    Prevents the systematic LLM length-undershoot failure mode (~0.8x of prompted
    target) by:
      1. Interpolating the CEILING (`max_words`) into the generation prompt rather
         than the floor (`min_words`), so a 0.85x model response lands squarely inside
         `[min_words, max_words]`.
      2. Measuring the generated output with the exact shared word counter and
         producing a deterministic concrete word-delta retry instruction when out of bounds.
    """

    def __init__(self, specs: Dict[str, LengthGateSpec] | None = None) -> None:
        self.specs = specs or dict(DEFAULT_TASK_LENGTH_SPECS)

    def build_generation_prompt(self, task_type: str, topic: str) -> str:
        spec = self.specs[task_type]
        return (
            f"Generate a '{task_type}' passage on topic '{topic}'. "
            f"Target length: {spec.max_words} words "
            f"(strict acceptance window: {spec.min_words} to {spec.max_words} words)."
        )

    def validate(self, task_type: str, generated_text: str) -> LengthValidationResult:
        spec = self.specs[task_type]
        wc = count_words(generated_text)
        midpoint = (spec.min_words + spec.max_words) // 2
        in_range = spec.min_words <= wc <= spec.max_words
        delta = midpoint - wc

        retry_instruction = None
        if not in_range:
            if wc < spec.min_words:
                needed = midpoint - wc
                retry_instruction = (
                    f"LENGTH_VIOLATION: Output contains {wc} words, which is below the "
                    f"minimum of {spec.min_words} words. Expand the passage by approximately "
                    f"+{needed} words to reach the {spec.min_words}-{spec.max_words} word window."
                )
            else:
                excess = wc - midpoint
                retry_instruction = (
                    f"LENGTH_VIOLATION: Output contains {wc} words, which exceeds the "
                    f"maximum of {spec.max_words} words. Condense the passage by approximately "
                    f"-{excess} words to fit within {spec.min_words}-{spec.max_words} words."
                )

        return LengthValidationResult(
            task_type=task_type,
            word_count=wc,
            min_words=spec.min_words,
            max_words=spec.max_words,
            prompted_ceiling_target=spec.max_words,
            in_range=in_range,
            word_delta_to_midpoint=delta,
            retry_feedback_instruction=retry_instruction,
        )
