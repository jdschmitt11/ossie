"""Canonical structural node identity for OSSIE-derived BSL models."""

from __future__ import annotations

import re
from typing import Any

import ibis
from boring_semantic_layer import SemanticModel
from boring_semantic_layer.ops import Dimension
from ossie_rules_engine import VENDOR

from .converter import NAMESPACE

IDENTITY_PREFIX = "evidence"
REQUIRED_EXTENSION_KEYS = ("evidence_id", "evidence_version", "anchor_node_type")

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _model_extensions(dimension: Dimension) -> dict[str, Any] | None:
    return (dimension.metadata or {}).get(NAMESPACE, {}).get("model_extensions")


def _find_anchor(model: SemanticModel) -> tuple[str, dict[str, Any]]:
    candidates = [
        (name, extensions)
        for name, dimension in model.get_dimensions().items()
        if (extensions := _model_extensions(dimension)) is not None
    ]
    if len(candidates) != 1:
        raise ValueError(
            "expected exactly one primary-key dimension carrying OSSIE model "
            f"extensions, found {len(candidates)}"
        )
    name, extensions = candidates[0]
    payload = extensions.get(VENDOR)
    if not isinstance(payload, dict):
        raise ValueError(f"anchor dimension {name!r} carries no {VENDOR} model extension")
    return name, payload


def _require(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"model extension is missing a non-empty {key!r}")
    return value


def with_node_identity(model: SemanticModel, *, evaluation_key: str | None) -> SemanticModel:
    """Add node identities, rejecting conflicting literal and upstream evaluation keys.

    ``evaluation_key`` is deterministic evaluation identity (never a physical
    partition column -- see rules_engine's ``EvaluationContext``/``PartitionRef``
    split). Pass ``evaluation_key=None`` to use an existing ``evaluation_key``
    dimension. Otherwise the non-empty literal is added as that dimension.
    Supplying both is invalid; neither value silently takes precedence.
    """
    source_evaluation_key = model.get_dimensions().get("evaluation_key")
    if source_evaluation_key is not None and evaluation_key is not None:
        raise ValueError(
            "evaluation_key conflict: the model already exposes evaluation_key; "
            "pass evaluation_key=None to use the upstream value"
        )
    if evaluation_key is None:
        if source_evaluation_key is None:
            raise ValueError(
                "evaluation_key must be a non-empty string unless the model exposes "
                "an explicit evaluation_key dimension"
            )
    elif not evaluation_key:
        raise ValueError("evaluation_key must be a non-empty string")

    anchor_name, payload = _find_anchor(model)
    evidence_id = _require(payload, "evidence_id")
    evidence_version = _require(payload, "evidence_version")
    anchor_node_type = _require(payload, "anchor_node_type")

    node_type_dimension = f"{anchor_node_type}_node_id"
    if not _IDENTIFIER.match(node_type_dimension):
        raise ValueError(
            f"anchor_node_type {anchor_node_type!r} does not produce a usable "
            f"dimension name ({node_type_dimension!r})"
        )

    identity_names = (
        "subject_node_id",
        "object_node_id",
        node_type_dimension,
        "evidence_node_id",
    )
    if evaluation_key is not None:
        identity_names = ("evaluation_key", *identity_names)
    existing = set(model.table.columns) | set(model.get_dimensions())
    collisions = sorted(name for name in identity_names if name in existing)
    if collisions:
        raise ValueError(
            "identity dimensions would overwrite existing names: " + ", ".join(collisions)
        )

    def anchor_string(table):
        return getattr(table, anchor_name).cast("string")

    def evaluation_key_string(table):
        if source_evaluation_key is not None:
            return source_evaluation_key(table).cast("string")
        return ibis.literal(evaluation_key)

    def build_evidence_node_id(table):
        return (
            f"{IDENTITY_PREFIX}::"
            + evaluation_key_string(table)
            + f"::{evidence_id}::{evidence_version}::"
            + anchor_string(table)
            + "::"
        )

    new_dimensions = {
        "subject_node_id": Dimension(expr=anchor_string, is_entity=True),
        "object_node_id": Dimension(expr=anchor_string, is_entity=True),
        node_type_dimension: Dimension(expr=anchor_string, is_entity=True),
        "evidence_node_id": Dimension(expr=build_evidence_node_id, is_entity=True),
    }
    if evaluation_key is not None:
        new_dimensions["evaluation_key"] = Dimension(
            expr=lambda table: ibis.literal(evaluation_key), is_entity=False
        )
    return model.with_dimensions(**new_dimensions)
