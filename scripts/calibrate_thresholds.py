"""Calibrate MAXSIM_THRESHOLD and CACHE_THRESHOLD by sweeping against labeled prompt pairs.

Run from the project root::

    python scripts/calibrate_thresholds.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.embedder import LocalEmbeddingEngine  # noqa: E402
from app.core.config import settings  # noqa: E402

import argparse
import pandas as pd

def load_dataset(file_path: str) -> list[dict[str, str | bool]]:
    """Loads prompt pairs from a CSV or Excel file."""
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
        
    try:
        if path.suffix == ".csv":
            df = pd.read_csv(path)
        elif path.suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        else:
            print(f"Error: Unsupported file format '{path.suffix}'. Use .csv or .xlsx")
            sys.exit(1)
            
        required_cols = {"prompt_a", "prompt_b", "should_cache"}
        if not required_cols.issubset(set(df.columns)):
            print(f"Error: File must contain columns: {', '.join(required_cols)}")
            sys.exit(1)
            
        # Ensure should_cache is boolean
        df["should_cache"] = df["should_cache"].astype(bool)
    
        # Convert to expected format
        dataset = []
        for _, row in df.iterrows():
            dataset.append({
                "a": str(row["prompt_a"]),
                "b": str(row["prompt_b"]),
                "should_cache": bool(row["should_cache"])
            })
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)


def cosine_distance(vec_a: list[float], vec_b: list[float]) -> float:
    """1 - cosine_similarity between two vectors."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0
    return float(1.0 - (np.dot(a, b) / denom))


def run_parameter_sweep(dataset: list[dict[str, str | bool]]) -> None:
    """Sweeps MAXSIM and cosine thresholds against the dataset, prints optimal values."""
    print(f"\nInitialising base embedder: {settings.EMBEDDING_MODEL_ID} ...")
    base_embedder = LocalEmbeddingEngine()

    print(f"Initialising ColBERT embedder: {settings.COLBERT_MODEL_ID} ...")
    try:
        from app.services.embedder import ColbertEmbeddingEngine
        embedder = ColbertEmbeddingEngine()
        colbert_enabled = True
    except Exception as e:
        print(f"[WARNING] Could not load ColBERT engine: {e}")
        embedder = None
        colbert_enabled = False

    print("\nCalculating embeddings and scores for dataset ...")
    results = []
    optimal_maxsim = 0.0

    for idx, item in enumerate(dataset):
        vec_a = base_embedder.get_embedding(str(item["a"]))
        vec_b = base_embedder.get_embedding(str(item["b"]))
        dist = cosine_distance(vec_a, vec_b)

        max_len = max(len(str(item["a"])), len(str(item["b"])))
        length_variance = abs(len(str(item["a"])) - len(str(item["b"]))) / max_len if max_len > 0 else 1.0

        score = -1.0
        if colbert_enabled and embedder is not None:
            a_matrix = embedder.get_colbert_embedding(str(item["a"]))
            b_matrix = embedder.get_colbert_embedding(str(item["b"]))
            score = embedder.compute_maxsim(a_matrix, b_matrix)

        results.append({
            "a": item["a"],
            "b": item["b"],
            "maxsim": score,
            "cosine_dist": dist,
            "length_variance": length_variance,
            "should_cache": item["should_cache"],
        })

    if colbert_enabled:
        print("\n--- RAW COLBERT MAXSIM SCORES ---")
        for r in results:
            status = "MATCH EXPECTED" if r["should_cache"] else "MISS EXPECTED "
            print(f"[{status}] MaxSim: {r['maxsim']:.2f} | {r['a'][:30]}... <-> {r['b'][:30]}...")

        print("\n--- MAXSIM THRESHOLD PARAMETER SWEEP ---")
        print(f"{'Threshold':<12} | {'True Positives':<16} | {'False Positives (FATAL)':<25} | {'F1-Score':<10}")
        print("-" * 70)

        best_f1_colbert = 0.0
        optimal_maxsim = 20.0
        for threshold in np.arange(10.0, 45.0, 0.5):
            tp = sum(1 for r in results if r["maxsim"] >= threshold and r["should_cache"])
            fp = sum(1 for r in results if r["maxsim"] >= threshold and not r["should_cache"])
            fn = sum(1 for r in results if r["maxsim"] < threshold and r["should_cache"])

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            marker = " <--- FATAL" if fp > 0 else ""

            if f1 > best_f1_colbert and fp == 0:
                best_f1_colbert = f1
                optimal_maxsim = threshold
            print(f"{threshold:.2f}         | {tp:<16} | {fp:<25} | {f1:.2f}{marker}")

    print("\n--- RAW COSINE DISTANCES (Fallback) ---")
    for r in results:
        status = "MATCH EXPECTED" if r["should_cache"] else "MISS EXPECTED "
        var_status = "PASS" if r['length_variance'] <= 0.25 else "FAIL"
        print(f"[{status}] Dist: {r['cosine_dist']:.3f} | Var: {r['length_variance']:.2f} ({var_status}) | {r['a'][:25]}...")

    print("\n--- CACHE_THRESHOLD PARAMETER SWEEP (with Length Variance <= 25%) ---")
    print(f"{'Threshold':<12} | {'True Positives':<16} | {'False Positives (FATAL)':<25} | {'F1-Score':<10}")
    print("-" * 70)

    best_f1_cosine = 0.0
    optimal_cosine = 0.12
    for threshold in np.arange(0.05, 0.35, 0.01):
        tp = sum(1 for r in results if r["cosine_dist"] <= threshold and r["length_variance"] <= 0.25 and r["should_cache"])
        fp = sum(1 for r in results if r["cosine_dist"] <= threshold and r["length_variance"] <= 0.25 and not r["should_cache"])
        fn = sum(1 for r in results if (r["cosine_dist"] > threshold or r["length_variance"] > 0.25) and r["should_cache"])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        marker = " <--- FATAL" if fp > 0 else ""

        if f1 > best_f1_cosine and fp == 0:
            best_f1_cosine = f1
            optimal_cosine = threshold
        print(f"{threshold:.2f}         | {tp:<16} | {fp:<25} | {f1:.2f}{marker}")

    print("\n" + "=" * 50)
    if colbert_enabled:
        print(f"OPTIMAL MAXSIM_THRESHOLD: {optimal_maxsim:.2f}")
    print(f"OPTIMAL CACHE_THRESHOLD (Cosine): {optimal_cosine:.2f}")
    print("=" * 50)
    print("Update these values in your .env file.")

def main():
    parser = argparse.ArgumentParser(description="Calibrate Semantic Cache Thresholds")
    parser.add_argument(
        "--file", 
        type=str, 
        default="assets/calibration_dataset.csv",
        help="Path to CSV or Excel file containing prompt_a, prompt_b, and should_cache columns."
    )
    args = parser.parse_args()

    print(f"Loading dataset from: {args.file}")
    dataset = load_dataset(args.file)
    print(f"Loaded {len(dataset)} prompt pairs.\n")
    
    run_parameter_sweep(dataset)

if __name__ == "__main__":
    main()
