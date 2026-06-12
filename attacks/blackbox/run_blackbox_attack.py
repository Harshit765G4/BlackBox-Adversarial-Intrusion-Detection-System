"""
attacks/blackbox/run_blackbox_attack.py

Entry point for running the black-box GA attack from the command line.

Usage:
    python -m attacks.blackbox.run_blackbox_attack
    python -m attacks.blackbox.run_blackbox_attack --seed 42
"""

import argparse
import json
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from attacks.blackbox.ga_engine import run_attack
from attacks.blackbox.oracle import EnsembleOracle
from core.config import GA_CONFIG, PATHS


def main():
    parser = argparse.ArgumentParser(description="Black-Box GA Attack")
    parser.add_argument("--seed",    type=int, default=None,   help="Attack seed index")
    parser.add_argument("--budget",  type=int, default=10_000, help="Query budget")
    parser.add_argument("--runs",    type=int, default=5,      help="Number of independent runs")
    parser.add_argument("--out",     type=str, default=None,   help="Save result JSON to path")
    args = parser.parse_args()

    all_results = []

    for run_i in range(args.runs):
        print(f"\n{'='*60}")
        print(f"RUN {run_i + 1}/{args.runs}")
        print('='*60)

        oracle = EnsembleOracle(query_limit=args.budget)
        result = run_attack(oracle=oracle, seed_idx=args.seed, verbose=True)

        print(f"\n{result.summary()}")
        all_results.append({
            "run":        run_i + 1,
            "success":    result.success,
            "generation": result.generation,
            "queries":    result.queries,
            "probability": result.probability,
            "perturbation": result.perturbation_norm,
            "realism_penalty": result.realism_penalty,
        })

    # Summary across runs
    successes = [r for r in all_results if r["success"]]
    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(successes)}/{len(all_results)} runs succeeded")
    if successes:
        avg_q = sum(r["queries"] for r in successes) / len(successes)
        avg_p = sum(r["probability"] for r in successes) / len(successes)
        print(f"  Avg queries (successful): {avg_q:.0f}")
        print(f"  Avg prob (successful):    {avg_p:.4f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()
