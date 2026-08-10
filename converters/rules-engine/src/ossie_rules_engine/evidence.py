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

"""Evidence metric YAML -> Ossie."""

from __future__ import annotations

import copy
import json

import yaml
from ossie import OSICustomExtension, OSIDataset, OSIDocument, OSISemanticModel
from sqlglot import exp, parse_one
from sqlglot.errors import ParseError

from ._common import (
    SOURCE_FORMAT,
    STASH_VERSION,
    VENDOR,
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


def convert_evidence_to_ossie(
    evidence_yaml: str, *, model_name: str, evidence_id: str
) -> str:
    """Convert an evidence metric YAML document into an Ossie YAML document."""
    document = load_yaml(evidence_yaml)
    filter_sql = document.get("filter")
    if filter_sql is not None:
        if not isinstance(filter_sql, str) or not filter_sql.strip():
            raise ConversionError("evidence YAML filter must be a non-empty SQL string")
    if "joins" in document:
        return _convert_joined_evidence(
            document, model_name=model_name, evidence_id=evidence_id
        )
    reject_unsupported({key: value for key, value in document.items() if key != "filter"})

    source = document.get("source")
    if not source:
        raise ConversionError("evidence YAML must declare a source")

    raw_fields = read_fields(document)
    raw_measures = read_measures(document)
    check_unique_names(raw_fields)
    check_unique_names(raw_measures)
    duplicate_names = {field["name"] for field in raw_fields} & {
        measure["name"] for measure in raw_measures
    }
    if duplicate_names:
        raise ConversionError(
            "fields and measures must have distinct names: "
            + ", ".join(sorted(duplicate_names))
        )

    fields = []
    anchors: list[str] = []
    anchor_tags: dict[str, str] = {}
    for raw in raw_fields:
        is_anchor, tags = parse_tags(raw.get("tags"), field_name=raw.get("name", "<unnamed>"))
        payload = dict(tags)
        if is_anchor:
            anchors.append(raw["name"])
            anchor_tags = tags
            payload = {"anchor": True, **tags}
        fields.append(
            build_field(
                raw,
                extensions=[extension(payload)] if payload else None,
                source=source,
            )
        )

    if len(anchors) != 1:
        raise ConversionError(
            f"evidence YAML must declare exactly one anchor field, found {len(anchors)}"
        )

    model_extensions = [
        extension(
            {
                "artifact_kind": "evidence",
                "source_format": SOURCE_FORMAT,
                "source_format_version": str(document.get("version", "")),
                "evidence_id": evidence_id,
                "evidence_version": anchor_tags.get("evidence_version"),
                "anchor_node_type": anchor_tags.get("node_type"),
            }
        )
    ]
    if filter_sql is not None:
        model_extensions.append(
            OSICustomExtension(
                vendor_name="DATABRICKS",
                data=json.dumps({"_v": STASH_VERSION, "filter": filter_sql}),
            )
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
        metrics=[build_metric(raw, source=source) for raw in raw_measures] or None,
        custom_extensions=model_extensions,
    )
    return OSIDocument(semantic_model=[model]).to_osi_yaml()


def _convert_joined_evidence(
    document: dict, *, model_name: str, evidence_id: str
) -> str:
    """Use the Databricks converter for UC joins, then attach Rules Engine tags."""
    try:
        from ossie_databricks import ConversionError as DatabricksConversionError
        from ossie_databricks import convert_metric_view_to_ossie
    except ImportError as exc:
        raise ConversionError(
            "joined evidence requires the apache-ossie-databricks converter"
        ) from exc

    source = document.get("source")
    if not source:
        raise ConversionError("evidence YAML must declare a source")

    raw_fields = read_fields(document)
    check_unique_names(raw_fields)
    anchor_name, anchor_tags, field_payloads = _field_metadata(raw_fields)
    _validate_join_metadata(document["joins"])

    metric_view = copy.deepcopy(document)
    sources = {model_name: source}
    metric_view["source"] = _as_select_source(source)

    relationship_metadata: dict[tuple[str, str], dict[str, str]] = {}

    def rewrite_join_sources(
        joins: list[dict], *, parent_alias: str, parent_dataset: str
    ) -> None:
        for join in joins:
            name = join.get("name")
            join_source = join.get("source")
            if not name or not join_source:
                raise ConversionError("every join must declare name and source")
            if name in sources:
                raise ConversionError(f"duplicate join name {name!r}")
            sources[name] = join_source
            metadata = _normalize_join_for_ossie(join, parent_alias=parent_alias)
            if metadata:
                relationship_metadata[(parent_dataset, name)] = metadata
            join["source"] = _as_select_source(join_source)
            rewrite_join_sources(
                join.get("joins", []), parent_alias=name, parent_dataset=name
            )

    rewrite_join_sources(
        metric_view["joins"], parent_alias="source", parent_dataset=model_name
    )
    try:
        converted = yaml.safe_load(
            convert_metric_view_to_ossie(yaml.safe_dump(metric_view), model_name=model_name)
        )
    except DatabricksConversionError as exc:
        raise ConversionError(f"Databricks Metric View conversion failed: {exc}") from exc

    model = converted["semantic_model"][0]
    datasets = {dataset["name"]: dataset for dataset in model["datasets"]}
    if set(datasets) != set(sources):
        raise ConversionError("Databricks conversion did not preserve the declared datasets")
    for name, dataset in datasets.items():
        dataset["source"] = sources[name]
    _attach_relationship_metadata(model, relationship_metadata)

    root = datasets[model_name]
    root["primary_key"] = [anchor_name]
    _attach_field_metadata(datasets, field_payloads)
    model.setdefault("custom_extensions", []).append(
        _extension_data(
            {
                "artifact_kind": "evidence",
                "source_format": SOURCE_FORMAT,
                "source_format_version": str(document.get("version", "")),
                "evidence_id": evidence_id,
                "evidence_version": anchor_tags.get("evidence_version"),
                "anchor_node_type": anchor_tags.get("node_type"),
            }
        )
    )
    return OSIDocument.model_validate(converted).to_osi_yaml()


def _as_select_source(source: str) -> str:
    """Adapt local fixture table names to the Databricks importer's source contract."""
    if source.lstrip().upper().startswith(("SELECT", "WITH")):
        return source
    return f"SELECT * FROM {source}"


def _validate_join_metadata(joins: list[dict]) -> None:
    """Reject join properties the Databricks importer cannot represent or preserve."""
    supported = {
        "name",
        "source",
        "on",
        "using",
        "joins",
        "rely",
        "cardinality",
        "edge_type",
    }
    for join in joins:
        unsupported = sorted(set(join) - supported)
        if unsupported:
            raise ConversionError(
                f"join {join.get('name', '<unnamed>')!r}: unsupported key(s): "
                + ", ".join(unsupported)
            )
        edge_type = join.get("edge_type")
        if edge_type is not None and (not isinstance(edge_type, str) or not edge_type.strip()):
            raise ConversionError(
                f"join {join.get('name', '<unnamed>')!r}: edge_type must be a non-empty string"
            )
        _validate_join_metadata(join.get("joins", []))


def _normalize_join_for_ossie(join: dict, *, parent_alias: str) -> dict[str, str]:
    """Separate a child-only join filter from an OSSIE equi-relationship."""
    name = join["name"]
    metadata = {}
    if "edge_type" in join:
        metadata["edge_type"] = join.pop("edge_type")
    if "using" in join:
        return metadata
    on_sql = join.get("on")
    if not on_sql:
        return metadata
    try:
        parsed = parse_one(on_sql, read="databricks")
    except ParseError as exc:
        raise ConversionError(f"join {name!r}: cannot parse on predicate {on_sql!r}") from exc
    conjuncts = _and_conjuncts(parsed)
    equi = [term for term in conjuncts if _is_parent_child_equality(term, parent_alias, name)]
    residual = [term for term in conjuncts if term not in equi]
    if not residual:
        return metadata
    if not equi:
        raise ConversionError(
            f"join {name!r}: residual predicates require at least one parent/child equality"
        )
    for term in residual:
        aliases = {column.table for column in term.find_all(exp.Column) if column.table}
        if aliases - {name}:
            raise ConversionError(
                f"join {name!r}: residual predicate must reference only {name!r}; "
                f"got {term.sql(dialect='databricks')!r}"
            )
    join["on"] = " AND ".join(term.sql(dialect="databricks") for term in equi)
    metadata["join_filter"] = " AND ".join(
        term.sql(dialect="databricks") for term in residual
    )
    return metadata


def _and_conjuncts(expression: exp.Expression) -> list[exp.Expression]:
    if isinstance(expression, exp.And):
        return _and_conjuncts(expression.this) + _and_conjuncts(expression.expression)
    return [expression]


def _is_parent_child_equality(
    expression: exp.Expression, parent_name: str, child_name: str
) -> bool:
    if not isinstance(expression, exp.EQ):
        return False
    if not isinstance(expression.this, exp.Column) or not isinstance(expression.expression, exp.Column):
        return False
    return {expression.this.table, expression.expression.table} == {parent_name, child_name}


def _attach_relationship_metadata(
    model: dict, metadata_by_connection: dict[tuple[str, str], dict[str, str]]
) -> None:
    for relationship in model.get("relationships", []):
        for (parent, child), metadata in metadata_by_connection.items():
            if {relationship["from"], relationship["to"]} == {parent, child}:
                relationship.setdefault("custom_extensions", []).append(_extension_data(metadata))
                break


def _field_metadata(
    raw_fields: list[dict],
) -> tuple[str, dict[str, str], dict[str, dict[str, object]]]:
    anchors: list[str] = []
    anchor_tags: dict[str, str] = {}
    payloads: dict[str, dict[str, object]] = {}
    for raw in raw_fields:
        name = raw.get("name")
        is_anchor, tags = parse_tags(raw.get("tags"), field_name=name or "<unnamed>")
        if is_anchor:
            anchors.append(name)
            anchor_tags = tags
            payloads[name] = {"anchor": True, **tags}
        elif tags:
            payloads[name] = tags
    if len(anchors) != 1:
        raise ConversionError(
            f"evidence YAML must declare exactly one anchor field, found {len(anchors)}"
        )
    return anchors[0], anchor_tags, payloads


def _attach_field_metadata(
    datasets: dict[str, dict], payloads: dict[str, dict[str, object]]
) -> None:
    fields = {
        field["name"]: field
        for dataset in datasets.values()
        for field in dataset.get("fields", [])
    }
    for name, payload in payloads.items():
        if name not in fields:
            raise ConversionError(
                f"Databricks conversion did not preserve tagged field {name!r}"
            )
        fields[name].setdefault("custom_extensions", []).append(_extension_data(payload))


def _extension_data(payload: dict[str, object]) -> dict[str, str]:
    return {
        "vendor_name": VENDOR,
        "data": json.dumps({"_v": STASH_VERSION, **payload}),
    }
