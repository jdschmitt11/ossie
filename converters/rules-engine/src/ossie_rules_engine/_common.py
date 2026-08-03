# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Shared helpers for the Rules Engine -> Ossie converter."""

from __future__ import annotations

import json
from typing import Any

import yaml
import sqlglot
from ossie import (
    OSICustomExtension,
    OSIDialect,
    OSIDialectExpression,
    OSIExpression,
    OSIField,
    OSIMetric,
)
from ossie_databricks._common import write_stash

VENDOR = "RULES_ENGINE"
"""``custom_extensions`` vendor name carrying Rules Engine metadata."""

STASH_VERSION = 1
"""Schema marker written as ``_v`` into every extension payload."""

SOURCE_FORMAT = "rules_engine_metric_yaml"

DIALECT = OSIDialect.DATABRICKS
"""Authored expressions use ``TRY_CAST``, which is not ANSI SQL."""

ANCHOR_TAG = "rules_engine:anchor"

# Field-level tags the converter understands. Anything else raises, so that
# authored metadata can never be silently dropped.
FIELD_TAG_KEYS = frozenset(
    {
        "node_type",
        "evidence_version",
        "semantic_role",
        "rule_id",
        "rule_severity",
        "rule_family_id",
        "rule_subject_columns",
        "rule_plain_english",
        "rule_version",
        "rule_audience_roles",
        "rule_applies_when",
    }
)

# Top-level and field-level constructs that are recognised but not yet
# implemented. They raise rather than being dropped.
UNSUPPORTED_TOP_LEVEL_KEYS = ("joins", "filter")
UNSUPPORTED_FIELD_KEYS = ("synonyms", "format", "edge_type", "joins", "source")


class ConversionError(Exception):
    """Raised when authored YAML cannot be converted without losing meaning."""


def load_yaml(text: str) -> dict[str, Any]:
    document = yaml.safe_load(text)
    if not isinstance(document, dict):
        raise ConversionError("metric YAML must be a mapping at the top level")
    return document


def extension(payload: dict[str, Any]) -> OSICustomExtension:
    """Build a ``RULES_ENGINE`` extension from a payload, adding the version marker."""
    return OSICustomExtension(
        vendor_name=VENDOR, data=json.dumps({"_v": STASH_VERSION, **payload})
    )


def parse_tags(tags: Any, *, field_name: str) -> tuple[bool, dict[str, str]]:
    """Split a field's ``tags`` list into its anchor marker and structured keys.

    ``rule_plain_english`` values contain their own punctuation, so tags are
    split on the first separator only.
    """
    if tags is None:
        return False, {}
    if not isinstance(tags, list):
        raise ConversionError(f"field {field_name!r}: tags must be a list")

    anchor = False
    parsed: dict[str, str] = {}
    for tag in tags:
        if not isinstance(tag, str) or ":" not in tag:
            raise ConversionError(f"field {field_name!r}: malformed tag {tag!r}")
        if tag == ANCHOR_TAG:
            anchor = True
            continue
        key, _, value = tag.partition(":")
        if key not in FIELD_TAG_KEYS:
            raise ConversionError(f"field {field_name!r}: unsupported tag {tag!r}")
        if key in parsed:
            raise ConversionError(f"field {field_name!r}: duplicate tag key {key!r}")
        parsed[key] = value
    return anchor, parsed


def reject_unsupported(document: dict[str, Any]) -> None:
    for key in UNSUPPORTED_TOP_LEVEL_KEYS:
        if key in document:
            raise ConversionError(f"{key!r} is not supported yet")


def read_fields(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the authored field list.

    Evidence artifacts spell this ``fields`` and rule families spell it
    ``dimensions``; they are the same construct.
    """
    present = [key for key in ("fields", "dimensions") if key in document]
    if not present:
        raise ConversionError("metric YAML must declare 'fields' or 'dimensions'")
    if len(present) > 1:
        raise ConversionError("metric YAML must declare only one of 'fields' or 'dimensions'")

    fields = document[present[0]]
    if not isinstance(fields, list) or not fields:
        raise ConversionError(f"{present[0]!r} must be a non-empty list")
    return fields


def read_measures(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Return optional UC Metric View measures."""
    measures = document.get("measures", [])
    if not isinstance(measures, list):
        raise ConversionError("'measures' must be a list")
    return measures


def _validate_expression_dependencies(
    expr: str, *, field_name: str, source: str
) -> None:
    """Reject ambient table reads from a single-source artifact expression."""
    try:
        tree = sqlglot.parse_one(f"SELECT {expr}", read="databricks")
    except sqlglot.errors.SqlglotError as exc:
        raise ConversionError(
            f"field {field_name!r}: expr is not valid Databricks SQL: {exc}"
        ) from exc

    cte_names = {
        cte.alias_or_name
        for cte in tree.find_all(sqlglot.exp.CTE)
        if cte.alias_or_name
    }
    allowed = {"source", source, *cte_names}
    undeclared = sorted(
        {
            table.name
            for table in tree.find_all(sqlglot.exp.Table)
            if table.name not in allowed
        }
    )
    if undeclared:
        raise ConversionError(
            f"field {field_name!r}: expr references undeclared table(s): "
            + ", ".join(undeclared)
        )


def build_field(
    raw: dict[str, Any], *, extensions: list[OSICustomExtension] | None, source: str
) -> OSIField:
    name = raw.get("name")
    if not name:
        raise ConversionError("every field must declare a name")
    for key in UNSUPPORTED_FIELD_KEYS:
        if key in raw:
            raise ConversionError(f"field {name!r}: {key!r} is not supported yet")
    expr = raw.get("expr")
    if not expr:
        raise ConversionError(f"field {name!r} must declare an expr")
    _validate_expression_dependencies(expr, field_name=name, source=source)

    return OSIField(
        name=name,
        expression=OSIExpression(
            dialects=[OSIDialectExpression(dialect=DIALECT, expression=expr)]
        ),
        label=raw.get("display_name"),
        description=raw.get("comment"),
        custom_extensions=extensions or None,
    )


def build_metric(raw: dict[str, Any], *, source: str) -> OSIMetric:
    """Convert one standard UC Metric View measure to an Ossie metric."""
    name = raw.get("name")
    expr = raw.get("expr")
    if not name:
        raise ConversionError("every measure must declare a name")
    if not expr:
        raise ConversionError(f"measure {name!r} must declare an expr")
    unsupported = sorted(
        set(raw)
        - {"name", "expr", "display_name", "comment", "format", "synonyms"}
    )
    if unsupported:
        raise ConversionError(
            f"measure {name!r}: unsupported key(s): " + ", ".join(unsupported)
        )
    _validate_expression_dependencies(expr, field_name=name, source=source)
    metric: dict[str, Any] = {
        "name": name,
        "expression": OSIExpression(
            dialects=[OSIDialectExpression(dialect=DIALECT, expression=expr)]
        ),
        "description": raw.get("comment"),
    }
    if "synonyms" in raw:
        metric["ai_context"] = {"synonyms": raw["synonyms"]}
    write_stash(metric, {"format": raw["format"]} if "format" in raw else {})
    return OSIMetric.model_validate(metric)


def check_unique_names(fields: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for raw in fields:
        name = raw.get("name")
        if name in seen:
            raise ConversionError(f"duplicate field name {name!r}")
        seen.add(name)
