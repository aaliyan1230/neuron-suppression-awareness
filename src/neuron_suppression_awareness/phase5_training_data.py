from __future__ import annotations

import random
from typing import Any

from .config import Phase5Config


def build_phase5_training_dataset(
    phase2a_train_rows: list[dict[str, Any]],
    config: Phase5Config,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    if rng is None:
        rng = random.Random(config.training.seed)

    n_base = len(phase2a_train_rows)
    fraction = config.suppression.fraction_final
    n_suppression = round(n_base * fraction / (1.0 - fraction))

    suppression_rows = []
    for _ in range(n_suppression):
        suppression_rows.append({
            "condition": "suppression_positive",
            "user_prompt": config.suppression.detection_prompt,
            "target_response": config.suppression.target_response,
            "concept": None,
            "vector_index": None,
            "alpha": None,
            "inject_noise": False,
            "mismatch_hint": None,
            "suppression_hook": True,
        })

    combined = list(phase2a_train_rows) + suppression_rows
    rng.shuffle(combined)

    for i, row in enumerate(combined):
        row["example_id"] = i

    return combined
