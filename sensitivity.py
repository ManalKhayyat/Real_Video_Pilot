"""Reproduce Table V from the pilot repository's saved pose features.

Run python sensitivity.py with NumPy installed. Each schedule repeats its
phase-observation mask every 2 seconds. The resulting CSV reports mean
absolute error in repetitions per set for each schedule and policy.
"""
import json
from decimal import Decimal, ROUND_HALF_UP
from itertools import product
from pathlib import Path
from statistics import mean

import numpy as np

from evaluate import MODELS, RATE, load, predict, write_csv

GAP_FRAMES = (2, 4, 6)
ONSET_FRAMES = (5, 10, 15, 20)
PERIOD_FRAMES = 2 * RATE


def mask_phase_observations(features, gap, onset):
    phase_angle, phase_quality, cue_geometry, cue_quality = features
    frame = np.arange(len(phase_angle))
    missing = (frame >= onset) & ((frame - onset) % PERIOD_FRAMES < gap)
    # The shared evaluator treats a non-finite phase angle as a missing pose.
    return np.where(missing, np.nan, phase_angle), phase_quality, cue_geometry, cue_quality


def run_schedules(clips, thresholds):
    for gap, onset in product(GAP_FRAMES, ONSET_FRAMES):
        masked_clips = [
            (clip, mask_phase_observations(clip["features"], gap, onset))
            for clip in clips
        ]
        for policy in MODELS:
            errors = []
            for clip, features in masked_clips:
                limits = thresholds[clip["exercise"]]
                count, *_ = predict(features, limits["bottom"], limits["top"], policy, "clean")
                errors.append(abs(count - int(clip["reference"])))
            yield {
                "seconds_masked": gap / RATE,
                "onset_seconds": onset / RATE,
                "policy": policy,
                "mae": float(mean(errors)),
            }


def format_mae(value):
    return str(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def main():
    root = Path(__file__).resolve().parent
    thresholds = json.loads((root / "frozen_thresholds.json").read_text(encoding="utf-8"))
    clips = load("evaluation")
    if len(clips) != 24 or any(int(clip["reference"]) != 10 for clip in clips):
        raise ValueError("Table V uses 24 evaluation sets with 10 reference repetitions each.")

    results = list(run_schedules(clips, thresholds))
    write_csv(root / "schedule_results.csv", results)
    print("Gap (s) | Voting proxies | Strict phase | Short-gap phase")
    for gap in GAP_FRAMES:
        duration = gap / RATE
        spans = []
        for policy in ("five_vote_proxy", "strict_phase", "gap_phase"):
            values = [row["mae"] for row in results
                      if row["seconds_masked"] == duration and row["policy"] == policy]
            spans.append(f"{format_mae(min(values))}–{format_mae(max(values))}")
        print(f"{duration:.1f} | " + " | ".join(spans))


if __name__ == "__main__":
    main()
