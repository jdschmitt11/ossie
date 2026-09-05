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

"""Rule family metric YAML -> Ossie."""

from __future__ import annotations

from typing import Any

from ossie import OSIDataset, OSIDocument, OSISemanticModel

from ._common import (
    SOURCE_FORMAT,
    ConversionError,
    build_field,
    build_metric,
    check_unique_names,
    extension,
    load_yaml,
    parse_tags,
    read_fields,
    read_measures,
    reject_unsupported,
)

RULE_ROLE = "rule_status"

#: Structured keys every ``rule_status`` dimension must carry.
REQUIRED_RULE_KEYS = ("rule_id", "rule_severity", "rule_family_id", "rule_plain_english")


#: Tags authored as a comma-separated list rather than a scalar string.
_CSV_KEYS = ("rule_subject_columns", "rule_audience_roles")

#: Tags authored as a comma-separated list of ``role=column`` pairs.
_MAPPING_KEYS = ("rule_observation_roles",)


def _mapping(value: str, *, key: str) -> dict[str, str]:
    """Parse ``role=column,role=column`` into a mapping, rejecting malformed pairs."""
    parsed: dict[str, str] = {}
    for entry in value.split(","):
        role, separator, column = entry.partition("=")
        if not separator or not role.strip() or not column.strip():
            raise ConversionError(f"{key}: malformed entry {entry!r}; expected role=column")
        if role.strip() in parsed:
            raise ConversionError(f"{key}: duplicate role {role.strip()!r}")
        parsed[role.strip()] = column.strip()
    return parsed


def _structure(tags: dict[str, str]) -> dict[str, Any]:
    """Apply per-key value shaping.

    Subject columns and audience roles are CSV lists; observation roles are a
    ``role=column`` mapping. Everything else stays the authored scalar.
    """
    payload: dict[str, Any] = dict(tags)
    for key in _CSV_KEYS:
        if key in payload:
            payload[key] = [value.strip() for value in payload[key].split(",")]
    for key in _MAPPING_KEYS:
        if key in payload:
            payload[key] = _mapping(payload[key], key=key)
    return payload


def convert_rules_to_ossie(rules_yaml: str, *, model_name: str) -> str:
    """Convert a rule family metric YAML document into an Ossie YAML document."""
    document = load_yaml(rules_yaml)
    reject_unsupported(document)

    source = document.get("source")
    if not source:
        raise ConversionError("rules YAML must declare a source")

    raw_fields = read_fields(document)
    check_unique_names(raw_fields)

    fields = []
    anchors: list[str] = []
    anchor_tags: dict[str, str] = {}
    rule_count = 0
    for raw in raw_fields:
        name = raw.get("name", "<unnamed>")
        is_anchor, tags = parse_tags(raw.get("tags"), field_name=name)
        if "rule_applies_when" in tags and not tags["rule_applies_when"].strip():
            raise ConversionError(f"rule {name!r}: rule_applies_when must not be empty")
        payload = _structure(tags)

        if tags.get("semantic_role") == RULE_ROLE:
            rule_count += 1
            missing = [key for key in REQUIRED_RULE_KEYS if key not in tags]
            if missing:
                raise ConversionError(
                    f"rule {name!r} is missing required tags: {', '.join(missing)}"
                )
            # rule_id is an independent identity (e.g. "MS-007"); it need not
            # equal the dimension name, matching the legacy loader
            # (ucmetric_loader.py:_rule_family_from_metric keeps rule_id and
            # name as separate RuleModel fields).

        if is_anchor:
            anchors.append(raw["name"])
            anchor_tags = tags
            payload = {"anchor": True, **payload}
        fields.append(
            build_field(
                raw,
                extensions=[extension(payload)] if payload else None,
                source=source,
            )
        )

    metrics = []
    for raw in read_measures(document):
        name = raw.get("name", "<unnamed>")
        raw_measure = dict(raw)
        measure_tags = raw_measure.pop("tags", None)
        is_anchor, tags = parse_tags(measure_tags, field_name=name)
        if is_anchor:
            raise ConversionError(f"measure {name!r} must not carry the anchor tag")
        metric = build_metric(raw_measure, source=source)
        if tags.get("semantic_role") == RULE_ROLE:
            rule_count += 1
            missing = [key for key in REQUIRED_RULE_KEYS if key not in tags]
            if missing:
                raise ConversionError(
                    f"rule {name!r} is missing required tags: {', '.join(missing)}"
                )
        if tags:
            metric = metric.model_copy(
                update={"custom_extensions": [extension(_structure(tags))]}
            )
        metrics.append(metric)

    if len(anchors) != 1:
        raise ConversionError(
            f"rules YAML must declare exactly one anchor dimension, found {len(anchors)}"
        )
    if rule_count == 0:
        raise ConversionError(
            "rules YAML must declare at least one rule_status dimension or measure"
        )

    model = OSISemanticModel(
        name=model_name,
        description=document.get("comment"),
        datasets=[
            OSIDataset(
                name=model_name,
                source=source,
                primary_key=anchors,
                fields=fields,
            )
        ],
        metrics=metrics or None,
        custom_extensions=[
            extension(
                {
                    "artifact_kind": "rules",
                    "source_format": SOURCE_FORMAT,
                    "source_format_version": str(document.get("version", "")),
                    "rule_family_id": model_name,
                    "rule_family_version": document.get("rule_family_version"),
                    "evidence_id": document.get("evidence_id"),
                    "evidence_version": document.get("evidence_version"),
                    "use_cases": document.get("use_cases"),
                    "render_hint": document.get("render_hint"),
                    "anchor_node_type": anchor_tags.get("node_type"),
                }
            )
        ],
    )
    return OSIDocument(semantic_model=[model]).to_osi_yaml()
