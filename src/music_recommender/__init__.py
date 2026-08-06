"""Hybrid music recommendation research package.

Public classes are loaded on first access.  This keeps lightweight modules such
as :mod:`music_recommender.cf` usable without importing pandas or scikit-learn.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "ContentRecommender", "HybridRecommender", "Interaction", "ItemKNN",
    "ImplicitALS", "MatrixFactorization", "MusicDataLoader", "ExperimentData",
    "MultiInterestContentRecommender", "PopularityRecommender", "Split",
    "SparseItemKNN", "leave_largest_out_split", "normalize_text",
]

_EXPORTS = {
    "Interaction": (".data", "Interaction"),
    "leave_largest_out_split": (".data", "leave_largest_out_split"),
    "normalize_text": (".data", "normalize_text"),
    "MusicDataLoader": (".data_loader", "MusicDataLoader"),
    "ExperimentData": (".data_loader", "ExperimentData"),
    "Split": (".data_loader", "Split"),
    "ContentRecommender": (".models", "ContentRecommender"),
    "HybridRecommender": (".models", "HybridRecommender"),
    "ItemKNN": (".models", "ItemKNN"),
    "MatrixFactorization": (".models", "MatrixFactorization"),
    "MultiInterestContentRecommender": (".models", "MultiInterestContentRecommender"),
    "PopularityRecommender": (".models", "PopularityRecommender"),
    "ImplicitALS": (".cf", "ImplicitALS"),
    "SparseItemKNN": (".cf", "SparseItemKNN"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
