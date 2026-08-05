#!/usr/bin/env python3
"""Build a versioned, train-fitted acoustic item feature matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / ".tools"
if TOOLS.exists():
    sys.path.insert(0, str(TOOLS))

try:
    import duckdb
except ImportError as exc:
    raise SystemExit("DuckDB is required; see data/databases/README.md") from exc

INTEGRATION = ROOT / "data" / "databases2" / "integration.duckdb"
OUTPUT_DIR = ROOT / "artifacts2" / "features"
CONTINUOUS = [
    "duration_ms",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]
BINARY = ["explicit", "mode"]
KEY_COLUMNS = [f"key_{value}" for value in range(12)] + ["key_unknown"]
TIME_SIGNATURE_COLUMNS = [
    "time_signature_3",
    "time_signature_4",
    "time_signature_5",
    "time_signature_other",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-version", default="feature_graph_u5_i2_v2")
    parser.add_argument(
        "--split-version", default="feature_split_u5_i2_eval20_seed42_v2"
    )
    parser.add_argument("--feature-schema-version", default="feature_matrix_audio_genre_v1")
    return parser.parse_args()


def ensure_registry(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS item_feature_schemas (
            feature_schema_version VARCHAR PRIMARY KEY,
            dataset_version VARCHAR NOT NULL REFERENCES feature_graph_datasets(dataset_version),
            split_version VARCHAR NOT NULL REFERENCES feature_split_datasets(split_version),
            item_count INTEGER NOT NULL CHECK (item_count > 0),
            feature_count INTEGER NOT NULL CHECK (feature_count > 0),
            artifact_path VARCHAR NOT NULL,
            artifact_sha256 VARCHAR NOT NULL,
            metadata_json VARCHAR NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT current_timestamp
        )
        """
    )


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def sql_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    versions = (
        args.dataset_version.strip(),
        args.split_version.strip(),
        args.feature_schema_version.strip(),
    )
    if not all(versions):
        raise SystemExit("dataset, split, and feature schema versions cannot be empty")

    con = duckdb.connect(str(INTEGRATION))
    con.execute("PRAGMA threads=4")
    ensure_registry(con)
    split_config = con.execute(
        """
        SELECT dataset_version
        FROM feature_split_datasets
        WHERE split_version = ?
        """,
        [args.split_version],
    ).fetchone()
    if split_config is None:
        raise SystemExit(f"unknown split version: {args.split_version}")
    if split_config[0] != args.dataset_version:
        raise SystemExit(
            f"split {args.split_version!r} belongs to {split_config[0]!r}, "
            f"not {args.dataset_version!r}"
        )

    existing = con.execute(
        """
        SELECT dataset_version, split_version, artifact_path, artifact_sha256
        FROM item_feature_schemas WHERE feature_schema_version = ?
        """,
        [args.feature_schema_version],
    ).fetchone()
    if existing is not None:
        if existing[:2] != (args.dataset_version, args.split_version):
            raise SystemExit(
                f"schema version {args.feature_schema_version!r} is already registered "
                f"for {existing[:2]}; choose a new version"
            )
        artifact = ROOT / existing[2]
        if artifact.exists() and file_sha256(artifact) == existing[3]:
            print(f"Item feature matrix already exists: {artifact}")
            con.close()
            return
        raise SystemExit(
            f"registered artifact is missing or changed: {artifact}; "
            "use a new feature schema version"
        )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW train_items AS
        SELECT DISTINCT feature_cluster_id
        FROM feature_dataset_splits
        WHERE split_version = {sql_string(args.split_version)} AND split = 'train'
        """
    )
    item_count = con.execute("SELECT count(*) FROM train_items").fetchone()[0]
    if not item_count:
        raise RuntimeError("the split has no training items")

    genre_values = con.execute(
            f"""
            SELECT DISTINCT c.track_genre
            FROM spotify_feature_clusters c
            JOIN train_items t USING (feature_cluster_id)
            WHERE c.track_genre IS NOT NULL AND c.track_genre <> ''
            ORDER BY c.track_genre
            """
        ).fetchall()
    
    genre_values = [row[0] for row in genre_values]
    if not genre_values:
        genre_values = ['unknown']

    genre_column_names = [f"genre_{genre}" for genre in genre_values] + ["genre_unknown"]
    genre_column_names_sql = [sql_ident(name) for name in genre_column_names]
    
    FEATURE_COLUMNS = (
        [f"{column}_z" for column in CONTINUOUS]
        + BINARY
        + KEY_COLUMNS
        + TIME_SIGNATURE_COLUMNS
        + genre_column_names_sql
    )

    stats: dict[str, dict[str, float | int]] = {}
    for column in CONTINUOUS:
        mean, standard_deviation, missing_count = con.execute(
            f"""
            SELECT avg(c.{column}), stddev_pop(c.{column}),
                   count(*) FILTER (WHERE c.{column} IS NULL)
            FROM spotify_feature_clusters c
            JOIN train_items t USING (feature_cluster_id)
            """
        ).fetchone()
        if mean is None or standard_deviation is None or standard_deviation == 0:
            raise RuntimeError(f"cannot scale constant or empty feature: {column}")
        stats[column] = {
            "mean": float(mean),
            "stddev_pop": float(standard_deviation),
            "missing_count": int(missing_count),
            "imputation": "train_mean",
        }

    scaled = []
    for column in CONTINUOUS:
        mean = stats[column]["mean"]
        stddev = stats[column]["stddev_pop"]
        scaled.append(
            f"(coalesce(c.{column}, {mean!r}) - {mean!r}) / "
            f"{stddev!r} AS {column}_z"
        )
    encoded = [
        "c.explicit::UTINYINT AS explicit",
        "c.mode::UTINYINT AS mode",
    ]
    encoded.extend(
        f"(c.key = {value})::UTINYINT AS key_{value}" for value in range(12)
    )
    encoded.append(
        "(c.key IS NULL OR c.key NOT BETWEEN 0 AND 11)::UTINYINT AS key_unknown"
    )
    encoded.extend(
        f"(c.time_signature = {value})::UTINYINT AS time_signature_{value}"
        for value in (3, 4, 5)
    )
    encoded.append(
        "(c.time_signature IS NULL OR c.time_signature NOT IN (3, 4, 5))"
        "::UTINYINT AS time_signature_other"
    )

    for genre, column_name in zip(genre_values, genre_column_names):
        encoded.append(
            f"(c.track_genre = {sql_string(genre)})::UTINYINT AS {sql_ident(column_name)}"
        )
    encoded.append(
        f"(c.track_genre IS NULL OR c.track_genre = '')::UTINYINT AS {sql_ident('genre_unknown')}"
    )


    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW item_feature_matrix AS
        SELECT
            c.feature_cluster_id,
            {", ".join(scaled + encoded)}
        FROM spotify_feature_clusters c
        JOIN train_items t USING (feature_cluster_id)
        ORDER BY c.feature_cluster_id
        """
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = OUTPUT_DIR / f"{args.feature_schema_version}.parquet"
    metadata_path = OUTPUT_DIR / f"{args.feature_schema_version}.metadata.json"
    temp_artifact = artifact.with_suffix(".parquet.tmp")
    temp_metadata = metadata_path.with_suffix(".json.tmp")
    con.execute(
        f"COPY item_feature_matrix TO '{sql_path(temp_artifact)}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )

    validation = con.execute(
        f"""
        SELECT
            count(*) AS row_count,
            count(DISTINCT feature_cluster_id) AS unique_items,
            count(*) FILTER (WHERE {
                " OR ".join(f"{column} IS NULL" for column in FEATURE_COLUMNS)
            }) AS rows_with_nulls,
            count(*) FILTER (WHERE {
                " + ".join(KEY_COLUMNS)
            } <> 1) AS invalid_key_encodings,
            count(*) FILTER (WHERE {
                " + ".join(TIME_SIGNATURE_COLUMNS)
            } <> 1) AS invalid_time_signature_encodings,
            count(*) FILTER (WHERE {
                " + ".join(genre_column_names_sql)
            } <> 1) AS invalid_genre_encodings
        FROM item_feature_matrix
        """
    ).fetchone()
    if validation != (item_count, item_count, 0, 0, 0, 0):
        raise RuntimeError(f"feature matrix integrity checks failed: {validation}")

    artifact_hash = file_sha256(temp_artifact)
    metadata = {
        "feature_schema_version": args.feature_schema_version,
        "dataset_version": args.dataset_version,
        "split_version": args.split_version,
        "fit_scope": "distinct feature_cluster_id values present in train",
        "item_count": item_count,
        "feature_count": len(FEATURE_COLUMNS),
        "id_column": "feature_cluster_id",
        "feature_columns": FEATURE_COLUMNS,
        "continuous_features": CONTINUOUS,
        "binary_features": BINARY,
        "categorical_encoding": {
            "key": KEY_COLUMNS,
            "time_signature": TIME_SIGNATURE_COLUMNS,
            "track_genre": genre_values,
        },
        "scaler": stats,
        "artifact": str(artifact.relative_to(ROOT)),
        "artifact_sha256": artifact_hash,
    }
    temp_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temp_artifact, artifact)
    os.replace(temp_metadata, metadata_path)

    metadata_json = json.dumps(metadata, sort_keys=True)
    con.execute(
        """
        INSERT INTO item_feature_schemas (
            feature_schema_version, dataset_version, split_version,
            item_count, feature_count, artifact_path, artifact_sha256,
            metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            args.feature_schema_version,
            args.dataset_version,
            args.split_version,
            item_count,
            len(FEATURE_COLUMNS),
            str(artifact.relative_to(ROOT)),
            artifact_hash,
            metadata_json,
        ],
    )
    con.execute("CHECKPOINT")
    con.close()

    print(f"Item feature matrix: {artifact}")
    print(f"  items: {item_count:,}")
    print(f"  encoded features: {len(FEATURE_COLUMNS):,}")
    print(f"  metadata: {metadata_path}")
    print(f"  sha256: {artifact_hash}")


if __name__ == "__main__":
    main()
