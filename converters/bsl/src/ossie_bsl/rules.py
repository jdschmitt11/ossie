"""Rule-status evaluation over an OSSIE-derived predicate model."""

from __future__ import annotations

from typing import Any

from boring_semantic_layer import SemanticModel
from boring_semantic_layer.ops import Dimension
from ossie_rules_engine import VENDOR

from .converter import NAMESPACE, applies_when_column

RULE_ROLE = "rule_status"
PASS = "PASS"
FAIL = "FAIL"
PENDING = "PENDING"
NOT_APPLICABLE = "NOT_APPLICABLE"
REQUIRED_KEYS = ("rule_id", "rule_severity", "rule_family_id", "rule_plain_english")


def _rule_metadata(dimension: Dimension) -> dict[str, Any] | None:
    extensions = (dimension.metadata or {}).get(NAMESPACE, {}).get("field_extensions", {})
    payload = extensions.get(VENDOR)
    if not isinstance(payload, dict) or payload.get("semantic_role") != RULE_ROLE:
        return None
    return payload


def _status(name: str, *, applies_when: bool, precomputed: bool = False):
    """Build the lazy status expression for one rule dimension.

    A boolean predicate maps to the PASS/FAIL/PENDING tri-state. A dimension
    tagged ``rule_status_encoding: precomputed`` already carries the complete
    status string, so it is projected as authored; coercing that VARCHAR
    through the boolean branch would fail outright. A null is PENDING in both
    encodings: the rule did not produce a verdict.
    """

    def build(table):
        column = getattr(table, name)
        if precomputed:
            status = column.isnull().ifelse(PENDING, column)
        else:
            # DuckDB's ``(predicate) IS TRUE`` coerces a non-boolean predicate
            # (an integer, or a CASE whose branches are all NULL) to boolean.
            # Match that rather than failing on a non-boolean column.
            predicate = column if column.type().is_boolean() else column.cast("boolean")
            status = predicate.isnull().ifelse(PENDING, predicate.ifelse(PASS, FAIL))
        if not applies_when:
            return status
        applicable = getattr(table, applies_when_column(name)).fill_null(False)
        return applicable.ifelse(status, NOT_APPLICABLE)

    return build


def evaluate_rules(predicate_model: SemanticModel) -> SemanticModel:
    """Replace boolean rule dimensions with lazy rule-status dimensions."""
    statuses: dict[str, Dimension] = {}
    for name, dimension in predicate_model.get_dimensions().items():
        metadata = _rule_metadata(dimension)
        if metadata is None:
            continue
        missing = [key for key in REQUIRED_KEYS if key not in metadata]
        if missing:
            raise ValueError(
                f"rule dimension {name!r} is missing required metadata: "
                f"{', '.join(missing)}"
            )
        statuses[name] = Dimension(
            expr=_status(
                name,
                applies_when="rule_applies_when" in metadata,
                precomputed=metadata.get("rule_status_encoding") == "precomputed",
            ),
            description=dimension.description,
            is_entity=dimension.is_entity,
            metadata=dimension.metadata,
        )

    if not statuses:
        raise ValueError(
            "predicate model carries no rule dimensions; expected at least one "
            f"dimension with {NAMESPACE!r} {VENDOR} metadata of role {RULE_ROLE!r}"
        )
    return predicate_model.with_dimensions(**statuses)
