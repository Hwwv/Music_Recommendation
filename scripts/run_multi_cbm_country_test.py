#!/usr/bin/env python3
"""Run the Multi-Interest Content Recommender on a sampled sub-population of one country.

This mirrors scripts/run_multi_cbm_test.py but restricts BOTH the training interactions
(used to build user profiles / cluster history) and the evaluation users to a
deterministic sample of users whose `country` matches --country. This tests
whether the content model does better on a more homogeneous population.

Run from the project root:
    python scripts/run_multi_cbm_country_test.py --country United States --country_sample_fraction 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TOOLS = ROOT / ".tools"
if TOOLS.exists():
    sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(SRC))

import duckdb

from music_recommender.data_loader import MusicDataLoader
from music_recommender.evaluation import evaluate_topk, assert_unseen_recommendations
from music_recommender.models import MultiInterestContentRecommender

INTEGRATION = ROOT / "data" / "databases" / "integration.duckdb"
LISTENING = ROOT / "data" / "databases" / "listening_clean.duckdb"
MULTI_CBM_OUTPUT_DIR = ROOT / "artifacts" / "multi_cbm_country_test"

DATASET_VERSION = "feature_graph_u5_i2_v1"
SPLIT_VERSION = "feature_split_u5_i2_eval20_seed42_v1"
FEATURE_SCHEMA_VERSION = "feature_matrix_audio_v1"

K = [20]
ALPHA = 0.9
GLOBAL_WEIGHT = 0.3
KMEANS = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_version", type=str, default=DATASET_VERSION)
    parser.add_argument("--split_version", type=str, default=SPLIT_VERSION)
    parser.add_argument("--feature_schema_version", type=str, default=FEATURE_SCHEMA_VERSION)
    parser.add_argument("--data_db_path", type=Path, default=INTEGRATION)
    parser.add_argument("--listening_db_path", type=Path, default=LISTENING)
    parser.add_argument("--multi_cbm_output_dir", type=str, default=MULTI_CBM_OUTPUT_DIR)
    parser.add_argument("--allow_test", default=True) # parser.add_argument("--allow_test", action="store_true")
    parser.add_argument("--k", type=int, nargs="+", default=K)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--global_weight", type=float, default=GLOBAL_WEIGHT)
    parser.add_argument("--k_for_kmeans", type=int,default=KMEANS)
    parser.add_argument("--continuous_only", action="store_true",
                         help="use only the leading continuous z-scored dims")
    parser.add_argument("--n_continuous", type=int, default=10)
    parser.add_argument("--country", type=str, default='United States',
                         help="exact country value as stored in listening_clean.duckdb users.country "
                              "(run scripts/check_country_codes.py first to confirm the spelling)")
    parser.add_argument("--country_sample_fraction", type=float, default=1)
    parser.add_argument("--country_seed", type=int, default=42)
    return parser.parse_args()


def get_country_sampled_users(
    data_db_path: Path,
    listening_db_path: Path,
    country: str,
    sample_fraction: float,
    seed: int,
) -> set[int]:
    """Deterministic sample of user_ids whose country matches `country`.

    Uses the same sha256-based deterministic shuffling pattern as
    build_feature_split.py so the sample is reproducible across runs.
    """
    con = duckdb.connect(str(data_db_path), read_only=True)
    con.execute(f"ATTACH '{listening_db_path}' AS listening (READ_ONLY)")
    rows = con.execute(
        """
        SELECT u.user_id
        FROM project_users u
        JOIN listening.users lu ON u.user_id = lu.user_id
        WHERE lu.country = ?
        ORDER BY sha256(concat(?::VARCHAR, ':', u.user_id::VARCHAR))
        """,
        [country, str(seed)],
    ).fetchall()
    con.execute("DETACH listening")
    con.close()
    all_ids = [r[0] for r in rows]
    if not all_ids:
        raise SystemExit(
            f"no users found with country == {country!r}; "
            "run scripts/check_country_codes.py to see the exact values stored"
        )
    cutoff = max(1, round(len(all_ids) * sample_fraction))
    return set(all_ids[:cutoff])


def main() -> None:
    args = parse_args()
    print(f"Running country-filtered Multi-Interest CBM: country={args.country!r} "
          f"fraction={args.country_sample_fraction} seed={args.country_seed}")

    sampled_users = get_country_sampled_users(
        data_db_path=args.data_db_path,
        listening_db_path=args.listening_db_path,
        country=args.country,
        sample_fraction=args.country_sample_fraction,
        seed=args.country_seed,
    )
    print(f"Sampled {len(sampled_users)} users with country={args.country!r} "
          f"({args.country_sample_fraction * 100:.0f}% of that population)")

    data_loader = MusicDataLoader(
        data_version=args.dataset_version,
        split_version=args.split_version,
        feature_schema_version=args.feature_schema_version,
        data_db_path=args.data_db_path,
        allow_test=args.allow_test,
    )

    train_interactions = data_loader.load_split_interactions("train")
    train_interactions = [row for row in train_interactions if row.user_id in sampled_users]
    print(f"Train interactions after country filter: {len(train_interactions):,}")

    features = data_loader.load_feature_mappings()
    if args.continuous_only:
        features = {item: vec[: args.n_continuous] for item, vec in features.items()}
        print(f"[diagnostic] using continuous-only features: {args.n_continuous} dims")

    test_truths = data_loader.load_split_truth("test")
    test_truths = {u: v for u, v in test_truths.items() if u in sampled_users}
    test_users = sorted(test_truths.keys())
    print(f"Test users after country filter: {len(test_users):,}")
    if not test_users:
        raise SystemExit("no test users left after filtering; try a larger --country_sample_fraction")

    alpha, gw, km = args.alpha, args.global_weight, args.k_for_kmeans
    print(f"alpha={alpha}, global weight={gw}, k fo kmeans={km}")
    multi_cbm_recommender = MultiInterestContentRecommender(confidence_alpha=alpha, global_weight=gw)
    multi_cbm_recommender.fit(train_interactions, features, k=km)

    seen = multi_cbm_recommender.seen.copy()
    catalog = multi_cbm_recommender.catalog

    recommendations = {}
    for user in test_users:
        recommendations[user] = multi_cbm_recommender.recommend(user, k=max(args.k))
    assert_unseen_recommendations(recommendations, seen)

    metrics = evaluate_topk(recommendations, test_truths, catalog, k_values=args.k)
    print(f"  metrics: {metrics}")

    output_dir = Path(args.multi_cbm_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "continuous_only" if args.continuous_only else "full"
    output_path = output_dir / f"cbm_{args.country}_{args.country_sample_fraction:.2f}_{suffix}.json"
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()