"""Calibrate CACHE_THRESHOLD using a dataset of positive and negative prompt pairs.

Run from the project root::

    python scripts/calibrate_thresholds.py

This script embeds every pair with the local ONNX model, computes cosine distances,
and sweeps thresholds from 0.05 to 0.35 to find the value that maximises F1-score
with zero false positives.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure the project root is on sys.path so ``app`` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.embedder import LocalEmbedder  # noqa: E402

# ---------------------------------------------------------------------------
# Calibration dataset
# ---------------------------------------------------------------------------
# Each entry is a pair of prompts.  ``should_cache=True`` means the two prompts
# are semantically equivalent and *should* produce a cache hit.
# ``should_cache=False`` means they have different intent and *must not* match.
DATASET: list[dict[str, str | bool]] = [
    # -- positive pairs (should match) -----------------------------------
    {
        "a": "What is the primary function of a reverse proxy?",
        "b": "Can you explain what reverse proxies do in a network?",
        "should_cache": True,
    },
    {
        "a": "Write a python function to add two integer numbers.",
        "b": "Show me how to add two integers together using Python.",
        "should_cache": True,
    },
    {
        "a": "How do I reset my corporate active directory password?",
        "b": "What are the steps to change my company login password?",
        "should_cache": True,
    },
    {
        "a": "What is the command to restart a running docker container?",
        "b": "How do I reboot a docker instance that is currently running?",
        "should_cache": True,
    },
    {
        "a": "Explain the difference between AWS S3 and AWS EBS storage.",
        "b": "Compare Amazon S3 object storage with EBS block storage.",
        "should_cache": True,
    },
    # -- negative pairs (must NOT match) ---------------------------------
    {
        "a": "Write a python function to add two integer numbers.",
        "b": "Write a python function to multiply two integer numbers.",
        "should_cache": False,
    },
    {
        "a": "How do I reset my corporate active directory password?",
        "b": "How do I reset my personal email password?",
        "should_cache": False,
    },
    {
        "a": "Drop the database table for user accounts.",
        "b": "Drop the database index for user accounts.",
        "should_cache": False,
    },
    {
        "a": "Start the production web server.",
        "b": "Stop the production web server.",
        "should_cache": False,
    },
    {
        "a": "Grant admin privileges to the guest user.",
        "b": "Revoke admin privileges from the guest user.",
        "should_cache": False,
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_distance(vec_a: list[float], vec_b: list[float]) -> float:
    """Return the cosine distance between two normalised vectors."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0  # orthogonal / zero vectors are maximally distant
    return float(1.0 - (np.dot(a, b) / denom))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_parameter_sweep() -> None:
    """Embed every pair, compute distances, and sweep thresholds."""
    print("\nInitialising ONNX embedder (offline mode) ...")
    embedder = LocalEmbedder()

    print("\nCalculating vector distances for dataset ...")
    results: list[dict] = []

    for item in DATASET:
        vec_a = embedder.get_embedding(item["a"])
        vec_b = embedder.get_embedding(item["b"])
        distance = cosine_distance(vec_a, vec_b)
        results.append({
            "a": item["a"],
            "b": item["b"],
            "distance": distance,
            "should_cache": item["should_cache"],
        })

    # -- raw distances -------------------------------------------------------
    print("\n--- RAW VECTOR DISTANCES ---")
    for r in results:
        status = "MATCH EXPECTED" if r["should_cache"] else "MISS EXPECTED "
        print(
            f"[{status}] Dist: {r['distance']:.3f} | "
            f"{r['a'][:30]}... <-> {r['b'][:30]}..."
        )

    # -- threshold sweep -----------------------------------------------------
    print("\n--- THRESHOLD PARAMETER SWEEP ---")
    print(f"{'Threshold':<12} | {'True Positives':<16} | {'False Positives (FATAL)':<25} | {'F1-Score':<10}")
    print("-" * 70)

    best_f1 = 0.0
    optimal_threshold = 0.05

    for threshold in np.arange(0.05, 0.36, 0.01):
        tp = sum(1 for r in results if r["distance"] <= threshold and r["should_cache"])
        fp = sum(1 for r in results if r["distance"] <= threshold and not r["should_cache"])
        fn = sum(1 for r in results if r["distance"] > threshold and r["should_cache"])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        marker = " <--- FATAL" if fp > 0 else ""

        if f1 > best_f1 and fp == 0:
            best_f1 = f1
            optimal_threshold = threshold

        print(f"{threshold:.2f}         | {tp:<16} | {fp:<25} | {f1:.2f}{marker}")

    print("\n" + "=" * 50)
    print(f"OPTIMAL CACHE_THRESHOLD: {optimal_threshold:.2f}")
    print("=" * 50)
    print("Update this value in your .env file.")
    print("Note: If you have False Positives at every threshold, your embedding model")
    print("      may be too small for your data. Consider a larger model.")


if __name__ == "__main__":
    run_parameter_sweep()
