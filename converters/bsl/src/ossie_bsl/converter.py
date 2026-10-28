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

"""Apache Ossie -> Boring Semantic Layer."""

from __future__ import annotations

import json
import operator
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import ibis
import yaml
from boring_semantic_layer import SemanticModel, to_semantic_table
from boring_semantic_layer.ops import CalcMeasure, Dimension, Measure
from ossie import (
    OSIDataset,
    OSIDialect,
    OSIDocument,
    OSIField,
    OSISemanticModel,
)
from sqlglot import exp, parse_one
from sqlglot.errors import ParseError

#: Alias the dataset source is exposed under inside the generated projection.
#: Ossie expressions produced by the Rules Engine converter qualify columns
#: with it.
SOURCE_ALIAS = "source"

#: Alias an upstream ``SemanticModel`` is bound to when chaining. It must differ
#: from ``SOURCE_ALIAS`` or the two stages collide as duplicate CTE names.
CHAIN_ALIAS = "rule_input"

#: Alias the ungrouped evidence relation is bound to inside an aggregate rule
#: source or a measure-authored status projection. Authored ``source.``
#: references are rebound to it for the same duplicate-CTE reason.
AGGREGATE_ALIAS = "rule_aggregate_input"

#: A second aggregation can follow an authored aggregate rule source when rule
#: statuses are measures. It needs its own alias so nested Ibis CTEs remain unique.
STATUS_AGGREGATE_ALIAS = "rule_status_aggregate_input"

#: Ibis transpiles the projection from this dialect. Authored expressions use
#: ``TRY_CAST``, which is not ANSI SQL.
SQL_DIALECT = "databricks"

#: Namespace for Ossie metadata inside ``Dimension.metadata``. BSL merges
#: dimension metadata flat into its JSON definition, so an un-namespaced key
#: such as ``description`` would clobber the built-in field.
NAMESPACE = "ossie"

#: Prefix for the physical predicate-projection column that carries a rule's
#: applicability condition. It is intentionally not a semantic dimension.
APPLIES_WHEN_PREFIX = "__ossie_applies_when__"

_RULES_ENGINE_VENDOR = "RULES_ENGINE"

_DIALECT_PREFERENCE = (OSIDialect.DATABRICKS, OSIDialect.ANSI_SQL)


class ConversionError(Exception):
    """Raised when an Ossie document cannot be represented as a BSL model."""


def _single_model(document: OSIDocument) -> OSISemanticModel:
    models = document.semantic_model
    if len(models) != 1:
        raise ConversionError(
            f"expected exactly one semantic model, found {len(models)}"
        )
    model = models[0]
    return model


def _single_dataset(model: OSISemanticModel) -> OSIDataset:
    datasets = model.datasets or []
    if len(datasets) != 1:
        raise ConversionError(f"expected exactly one dataset, found {len(datasets)}")
    dataset = datasets[0]
    if not dataset.fields:
        raise ConversionError(f"dataset {dataset.name!r} declares no fields")
    return dataset


def _pick_expression(field: Any) -> str:
    """Select the best available dialect, preferring DATABRICKS over ANSI_SQL."""
    by_dialect = {entry.dialect: entry.expression for entry in field.expression.dialects}
    for dialect in _DIALECT_PREFERENCE:
        if dialect in by_dialect:
            return by_dialect[dialect]
    available = ", ".join(sorted(entry.value for entry in by_dialect))
    raise ConversionError(
        f"field {field.name!r} has no DATABRICKS or ANSI_SQL expression (has: {available})"
    )


def _extensions(
    obj: OSIDataset | OSIField | OSISemanticModel,
) -> dict[str, Any]:
    return {
        extension.vendor_name: json.loads(extension.data)
        for extension in (obj.custom_extensions or [])
    }


def applies_when_column(rule_name: str) -> str:
    """Return the reserved physical column name for a rule precondition."""
    return f"{APPLIES_WHEN_PREFIX}{rule_name}"


def _rule_applies_when(field: OSIField) -> str | None:
    payload = _extensions(field).get(_RULES_ENGINE_VENDOR)
    if not isinstance(payload, dict):
        return None
    applies_when = payload.get("rule_applies_when")
    if applies_when is None:
        return None
    if not isinstance(applies_when, str) or not applies_when.strip():
        raise ConversionError(
            f"field {field.name!r}: rule_applies_when must be a non-empty SQL string"
        )
    return applies_when


def _applies_when_columns(dataset: OSIDataset) -> dict[str, str]:
    return {
        field.name: applies_when_column(field.name)
        for field in dataset.fields
        if _rule_applies_when(field) is not None
    }


def _projection_sql(
    dataset: OSIDataset,
    *,
    alias: str,
    fields: list[OSIField] | None = None,
    metrics: list[Any] | None = None,
) -> str:
    selected_fields = list(dataset.fields if fields is None else fields)
    selections = [f"{_pick_expression(field)} AS {field.name}" for field in selected_fields]
    selections.extend(
        f"{_rule_applies_when(field)} AS {applies_when_column(field.name)}"
        for field in selected_fields
        if _rule_applies_when(field) is not None
    )
    selections.extend(
        f"{_metric_argument_expression(metric, dataset=dataset)} "
        f"AS {_metric_argument_name(metric.name)}"
        for metric in (metrics or [])
        if _base_metric(metric) is not None
    )
    selections.insert(0, f"{SOURCE_ALIAS}.*")
    columns = ", ".join(selections)
    return f"SELECT {columns} FROM {alias} AS {SOURCE_ALIAS}"


def _source_passthrough_field(field: OSIField) -> bool:
    """Whether a chained rule field reuses an identically named input dimension."""
    try:
        expression = parse_one(_pick_expression(field), read=SQL_DIALECT)
    except ParseError as exc:
        raise ConversionError(f"cannot parse field expression for {field.name!r}") from exc
    return (
        isinstance(expression, exp.Column)
        and expression.table == SOURCE_ALIAS
        and expression.name == field.name
    )


@dataclass(frozen=True)
class RuleSourceSpec:
    """The shape of an authored aggregate rule source.

    ``dimensions`` are the bare GROUP BY columns, ``measures`` the names read
    through ``MEASURE(name) AS name``, and ``aggregates`` every other item as
    ``(alias, sql)``: a raw aggregate expression evaluated in the same GROUP BY.
    """

    dimensions: tuple[str, ...]
    measures: tuple[str, ...]
    aggregates: tuple[tuple[str, str], ...]


def _quoted(name: str) -> str:
    return exp.to_identifier(name, quoted=True).sql(dialect=SQL_DIALECT)


def _measure_call_name(node: exp.Expression) -> str | None:
    """Return ``name`` for a ``MEASURE(name)`` call, else ``None``."""
    if not isinstance(node, exp.Anonymous) or node.name.upper() != "MEASURE":
        return None
    if len(node.expressions) != 1:
        raise ConversionError("rule source supports only MEASURE(name) with one argument")
    argument = node.expressions[0]
    if not isinstance(argument, exp.Column) or argument.table:
        raise ConversionError("MEASURE() must name one unqualified evidence measure")
    return argument.name


def _aggregate_item_sql(item: exp.Alias) -> str:
    """Validate a raw aggregate SELECT item and render it against ``source``."""
    expression = item.this
    if expression.find(exp.AggFunc) is None:
        raise ConversionError(
            f"rule source item {item.alias!r} must be an aggregate expression or MEASURE(name)"
        )
    if any(_measure_call_name(node) is not None for node in expression.find_all(exp.Anonymous)):
        raise ConversionError(
            f"rule source item {item.alias!r} may not nest MEASURE() inside another expression"
        )

    def bind(node: exp.Expression) -> exp.Expression:
        if not isinstance(node, exp.Column):
            return node
        if node.table and node.table != SOURCE_ALIAS:
            raise ConversionError(
                f"rule source item {item.alias!r} references unknown dataset {node.table!r}"
            )
        return exp.column(node.name, table=AGGREGATE_ALIAS)

    return expression.transform(bind).sql(dialect=SQL_DIALECT)


def parse_rule_source(source: str) -> RuleSourceSpec:
    """Validate the aggregate query shape used as a rule source."""
    try:
        query = parse_one(source, read=SQL_DIALECT)
    except ParseError as exc:
        raise ConversionError(f"cannot parse rule measure source {source!r}") from exc
    if not isinstance(query, exp.Select):
        raise ConversionError("rule measure source must be a SELECT query")
    if query.args.get("with_") or query.args.get("joins") or query.args.get("where"):
        raise ConversionError(
            "rule measure source supports one metric-view FROM with no WITH, JOIN, or WHERE"
        )
    from_clause = query.args.get("from_")
    if not from_clause or not isinstance(from_clause.this, exp.Table):
        raise ConversionError("rule measure source must select from exactly one metric view")
    if query.args.get("having") or query.args.get("order") or query.args.get("limit"):
        raise ConversionError(
            "rule measure source does not support HAVING, ORDER BY, or LIMIT"
        )

    dimensions: list[str] = []
    measures: list[str] = []
    aggregates: list[tuple[str, str]] = []
    for item in query.expressions:
        if isinstance(item, exp.Column) and not item.table:
            dimensions.append(item.name)
            continue
        if not isinstance(item, exp.Alias):
            raise ConversionError(
                "rule measure source SELECT items must be bare dimensions, "
                "MEASURE(name) AS name, or <aggregate> AS alias"
            )
        measure_name = _measure_call_name(item.this)
        if measure_name is not None:
            if item.alias != measure_name:
                raise ConversionError("MEASURE(name) must be aliased to the same name")
            measures.append(measure_name)
            continue
        aggregates.append((item.alias, _aggregate_item_sql(item)))
    outputs = [*dimensions, *measures, *(alias for alias, _ in aggregates)]
    if not dimensions or len(outputs) == len(dimensions) or len(set(outputs)) != len(outputs):
        raise ConversionError(
            "rule measure source requires unique dimensions and at least one "
            "MEASURE(name) or aggregate item"
        )
    group = query.args.get("group")
    group_names = [
        item.name
        for item in (group.expressions if group is not None else [])
        if isinstance(item, exp.Column) and not item.table
    ]
    if group_names != dimensions:
        raise ConversionError(
            "rule measure source GROUP BY must list the selected dimensions in order"
        )
    return RuleSourceSpec(tuple(dimensions), tuple(measures), tuple(aggregates))


def rule_source_measure_names(source: str) -> list[str]:
    """Names an authored rule source reads through ``MEASURE(name)``."""
    return list(parse_rule_source(source).measures)


def _dimension_relation(source_model: SemanticModel) -> Any:
    """The ungrouped upstream relation with every dimension as a column."""
    table = source_model.table
    dimensions = source_model.get_dimensions()
    return table.select(**{name: dimension(table) for name, dimension in dimensions.items()})


def _chain_base(source_model: SemanticModel) -> Any:
    """The upstream relation with every dimension materialised as a column.

    A BSL model may declare a dimension as an expression over its table; the
    chained projection reads dimensions by name, so each is computed here.
    Physical columns that back measures are kept for MEASURE() reads.
    """
    table = source_model.table
    dimensions = source_model.get_dimensions()
    return table.mutate(**{name: dimension(table) for name, dimension in dimensions.items()})


def _aggregate_relation(
    source_model: SemanticModel,
    *,
    group: list[str],
    measures: list[str],
    aggregates: list[tuple[str, str]],
    aggregate_alias: str = AGGREGATE_ALIAS,
) -> Any:
    """One row per ``group``: BSL measures joined with raw aggregate items."""
    relation = None
    if measures:
        relation = source_model.query(dimensions=group, measures=measures).to_untagged()
    if aggregates:
        selections = [*(_quoted(name) for name in group)]
        selections.extend(f"{sql} AS {_quoted(alias)}" for alias, sql in aggregates)
        raw = _dimension_relation(source_model).alias(aggregate_alias).sql(
            f"SELECT {', '.join(selections)} FROM {aggregate_alias} "
            f"GROUP BY {', '.join(_quoted(name) for name in group)}",
            dialect=SQL_DIALECT,
        )
        relation = raw if relation is None else relation.join(raw, group)
    return relation


def _structural_dimensions(source_model: SemanticModel, *, selected: list[str]) -> list[str]:
    return [
        name
        for name, dimension in source_model.get_dimensions().items()
        if dimension.is_entity and name not in selected
    ]


def _anchor_source_columns(dataset: OSIDataset) -> list[str]:
    """Columns the dataset's primary-key fields read from their source.

    A rules document may name its anchor differently from the evidence it
    chains onto (``plan_id: source.sk_plan``). What must survive an aggregate
    rule source is the column the anchor is derived from, not the anchor's own
    name, so resolve each primary-key field to the columns its expression
    reads.
    """
    by_name = {field.name: field for field in dataset.fields}
    columns: list[str] = []
    for name in dataset.primary_key or []:
        field = by_name.get(name)
        if field is None:
            columns.append(name)
            continue
        try:
            expression = parse_one(_pick_expression(field), read=SQL_DIALECT)
        except ParseError as exc:
            raise ConversionError(f"cannot parse anchor expression for {name!r}") from exc
        referenced = [column.name for column in expression.find_all(exp.Column)]
        columns.extend(referenced or [name])
    return list(dict.fromkeys(columns))


def query_rule_source_model(
    source_model: SemanticModel,
    *,
    source: str,
    primary_key: list[str] | None = None,
    measure_columns: Iterable[str] = (),
) -> SemanticModel:
    """Evaluate a declared aggregate rule-source query lazily through BSL.

    ``MEASURE(name)`` reads a measure of ``source_model``. When the evidence
    has already been materialised, a former measure is a plain column; the
    caller lists such columns in ``measure_columns`` to vouch that each is
    constant within every group, and the converter reads it with ``MAX``.
    """
    spec = parse_rule_source(source)
    upstream_dimensions = source_model.get_dimensions()
    missing = sorted(set(spec.dimensions).difference(upstream_dimensions))
    if missing:
        raise ConversionError(
            "rule measure source references unknown evidence dimensions: " + ", ".join(missing)
        )
    group = [*spec.dimensions, *_structural_dimensions(source_model, selected=list(spec.dimensions))]
    missing_anchor = sorted(set(primary_key or []).difference(group))
    if missing_anchor:
        raise ConversionError(
            "rule measure source drops the column(s) its anchor is derived from: "
            + ", ".join(missing_anchor)
        )
    known_measures = set(source_model.get_measures()) | set(
        source_model.get_calculated_measures()
    )
    constant_columns = set(measure_columns)
    measures: list[str] = []
    aggregates = list(spec.aggregates)
    for name in spec.measures:
        if name in known_measures:
            measures.append(name)
        elif name in constant_columns and name in upstream_dimensions:
            aggregates.append((name, f"MAX({AGGREGATE_ALIAS}.{_quoted(name)})"))
        else:
            raise ConversionError(
                f"MEASURE({name}) names neither a measure of the source model nor a "
                "column declared constant per group in measure_columns"
            )
    for alias, sql in aggregates:
        unknown = sorted(
            {
                column.name
                for column in parse_one(sql, read=SQL_DIALECT).find_all(exp.Column)
                if column.name not in upstream_dimensions
            }
        )
        if unknown:
            raise ConversionError(
                f"rule source item {alias!r} references unknown evidence dimensions: "
                + ", ".join(unknown)
            )
    relation = _aggregate_relation(
        source_model, group=group, measures=measures, aggregates=aggregates
    )
    return _rebind_semantic_relation(relation, source_model=source_model)


def _rules_engine_payload(obj: Any) -> dict[str, Any]:
    payload = _extensions(obj).get(_RULES_ENGINE_VENDOR)
    return payload if isinstance(payload, dict) else {}


def _subject_columns(payload: Mapping[str, Any]) -> list[str]:
    raw = payload.get("rule_subject_columns") or []
    if isinstance(raw, str):
        raw = raw.split(",")
    return [str(column).strip() for column in raw if str(column).strip()]


def _bound_metric_sql(
    metric: Any, *, dataset: OSIDataset, aggregate_alias: str = AGGREGATE_ALIAS
) -> str:
    """Render a metric expression against the ``source`` alias."""
    try:
        parsed = parse_one(_pick_expression(metric), read=SQL_DIALECT)
    except ParseError as exc:
        raise ConversionError(f"cannot parse metric {metric.name!r}") from exc

    def bind(node: exp.Expression) -> exp.Expression:
        if not isinstance(node, exp.Column):
            return node
        if node.table and node.table not in {SOURCE_ALIAS, dataset.name}:
            raise ConversionError(
                f"metric {metric.name!r} references unknown dataset {node.table!r}"
            )
        return exp.column(node.name, table=aggregate_alias)

    return parsed.transform(bind).sql(dialect=SQL_DIALECT)


def _bound_metric_applies_when_sql(
    metric: Any, *, dataset: OSIDataset, aggregate_alias: str
) -> str | None:
    expression = _rule_applies_when(metric)
    if expression is None:
        return None
    try:
        parsed = parse_one(expression, read=SQL_DIALECT)
    except ParseError as exc:
        raise ConversionError(
            f"cannot parse rule_applies_when for metric {metric.name!r}"
        ) from exc

    def bind(node: exp.Expression) -> exp.Expression:
        if not isinstance(node, exp.Column):
            return node
        if node.table and node.table not in {SOURCE_ALIAS, dataset.name}:
            raise ConversionError(
                f"metric {metric.name!r} rule_applies_when references unknown "
                f"dataset {node.table!r}"
            )
        return exp.column(node.name, table=aggregate_alias)

    return parsed.transform(bind).sql(dialect=SQL_DIALECT)


def _aggregate_status_model(
    source_model: SemanticModel,
    *,
    dataset: OSIDataset,
    metrics: list[Any],
) -> SemanticModel:
    """Collapse the evidence to one row per anchor for measure-authored statuses.

    Every metric is evaluated in one GROUP BY over the anchor (plus any other
    upstream entity dimension); each rule's subject columns ride along as
    ``MAX(column)`` so the observed values stay available per subject.
    """
    upstream_dimensions = source_model.get_dimensions()
    anchors: list[str] = []
    for field in dataset.fields:
        if field.name not in (dataset.primary_key or []):
            continue
        if field.name not in upstream_dimensions or not _source_passthrough_field(field):
            raise ConversionError(
                f"aggregate rule statuses require anchor {field.name!r} to pass through "
                "an identically named evidence dimension"
            )
        anchors.append(field.name)
    if not anchors:
        raise ConversionError("aggregate rule statuses require a primary key anchor")
    group = [*anchors, *_structural_dimensions(source_model, selected=anchors)]
    metric_names = {metric.name for metric in metrics}
    aggregates: list[tuple[str, str]] = []
    carried: list[str] = []
    for metric in metrics:
        for column in _subject_columns(_rules_engine_payload(metric)):
            if column in metric_names or column in group or column in carried:
                continue
            if column not in upstream_dimensions:
                raise ConversionError(
                    f"metric {metric.name!r} names unknown subject column {column!r}"
                )
            carried.append(column)
    aggregates.extend(
        (column, f"MAX({STATUS_AGGREGATE_ALIAS}.{_quoted(column)})")
        for column in carried
    )
    aggregates.extend(
        (
            metric.name,
            _bound_metric_sql(
                metric, dataset=dataset, aggregate_alias=STATUS_AGGREGATE_ALIAS
            ),
        )
        for metric in metrics
    )
    aggregates.extend(
        (applies_when_column(metric.name), expression)
        for metric in metrics
        if (
            expression := _bound_metric_applies_when_sql(
                metric,
                dataset=dataset,
                aggregate_alias=STATUS_AGGREGATE_ALIAS,
            )
        )
        is not None
    )
    relation = _aggregate_relation(
        source_model,
        group=group,
        measures=[],
        aggregates=aggregates,
        aggregate_alias=STATUS_AGGREGATE_ALIAS,
    )
    return _rebind_semantic_relation(relation, source_model=source_model)


def rebind_rule_source_model(source_model: SemanticModel) -> SemanticModel:
    """Expose declared evidence dimensions as a lazy rule-source relation."""
    dimensions = source_model.get_dimensions()
    relation = source_model.query(dimensions=list(dimensions)).to_untagged()
    return _rebind_semantic_relation(relation, source_model=source_model)


def _rebind_semantic_relation(relation: Any, *, source_model: SemanticModel) -> SemanticModel:
    """Wrap a BSL query relation while retaining upstream dimension metadata."""
    upstream_dimensions = source_model.get_dimensions()
    queried = to_semantic_table(relation, name=source_model.name)
    queried_dimensions = {
        name: Dimension(
            expr=_reference(name),
            is_entity=bool(getattr(upstream_dimensions.get(name), "is_entity", False)),
            metadata=dict(getattr(upstream_dimensions.get(name), "metadata", {}) or {}),
        )
        for name in relation.columns
    }
    return queried.with_dimensions(**queried_dimensions)


def _reference(name: str):
    """Build a column accessor bound to ``name`` rather than the loop variable."""
    return lambda table: getattr(table, name)


_METRIC_ARGUMENT_PREFIX = "__ossie_metric_argument__"
_AGGREGATE_METHODS = {
    exp.Min: "min",
    exp.Max: "max",
    exp.Sum: "sum",
    exp.Avg: "mean",
    exp.Count: "count",
}
_CALCULATED_BINARY_OPERATORS = {
    exp.Add: operator.add,
    exp.Sub: operator.sub,
    exp.Mul: operator.mul,
    exp.Div: operator.truediv,
    exp.EQ: operator.eq,
    exp.NEQ: operator.ne,
    exp.LT: operator.lt,
    exp.LTE: operator.le,
    exp.GT: operator.gt,
    exp.GTE: operator.ge,
    exp.And: operator.and_,
    exp.Or: operator.or_,
}


def _metric_argument_name(metric_name: str) -> str:
    return f"{_METRIC_ARGUMENT_PREFIX}{metric_name}"


def _base_metric(metric: Any) -> tuple[str, exp.Expression] | None:
    """Parse a base aggregate into its Ibis reduction and SQL argument."""
    try:
        parsed = parse_one(_pick_expression(metric), read=SQL_DIALECT)
    except ParseError as exc:
        raise ConversionError(f"cannot parse metric {metric.name!r}") from exc
    if type(parsed) not in _AGGREGATE_METHODS:
        return None
    argument = parsed.this
    method = _AGGREGATE_METHODS[type(parsed)]
    if isinstance(parsed, exp.Count) and isinstance(argument, exp.Distinct):
        if len(argument.expressions) != 1:
            raise ConversionError(
                f"metric {metric.name!r}: COUNT DISTINCT requires one argument"
            )
        return "nunique", argument.expressions[0]
    if argument is None or isinstance(argument, exp.Star):
        raise ConversionError(
            f"metric {metric.name!r} must aggregate one expression, not *"
        )
    return method, argument


def _metric_argument_expression(
    metric: Any,
    *,
    dataset: OSIDataset,
) -> str:
    base = _base_metric(metric)
    if base is None:
        raise ConversionError(f"metric {metric.name!r} is not a base aggregate")
    argument = base[1]

    def bind_dataset(node: exp.Expression) -> exp.Expression:
        if not isinstance(node, exp.Column):
            return node
        if node.table not in {SOURCE_ALIAS, dataset.name}:
            raise ConversionError(
                f"metric {metric.name!r} references unknown dataset {node.table!r}"
            )
        return exp.column(node.name, table=SOURCE_ALIAS)

    return argument.transform(bind_dataset).sql(dialect=SQL_DIALECT)


def _measure(metric: Any, *, method: str) -> Measure:
    column = _metric_argument_name(metric.name)
    return Measure(
        expr=lambda table, _column=column, _method=method: getattr(
            table[_column], _method
        )(),
        description=metric.description,
    )


def _calculated_measure(metric: Any, *, measure_names: set[str]) -> CalcMeasure:
    """Build a BSL post-aggregate measure over named measure dependencies."""
    try:
        parsed = parse_one(_pick_expression(metric), read=SQL_DIALECT)
    except ParseError as exc:
        raise ConversionError(f"cannot parse calculated metric {metric.name!r}") from exc
    columns = list(parsed.find_all(exp.Column))
    qualified = sorted(column.sql() for column in columns if column.table)
    if qualified:
        raise ConversionError(
            f"calculated metric {metric.name!r} must reference measures by name, not "
            + ", ".join(qualified)
        )
    dependencies = frozenset(column.name for column in columns)
    unknown = sorted(dependencies - measure_names)
    if not dependencies or unknown:
        detail = "no measure dependencies" if not dependencies else ", ".join(unknown)
        raise ConversionError(
            f"calculated metric {metric.name!r} references {detail}"
        )

    def calculated(scope):
        def resolve(node: exp.Expression) -> Any:
            if isinstance(node, exp.Column):
                return getattr(scope, node.name)
            if isinstance(node, exp.Literal):
                return node.to_py()
            if isinstance(node, exp.Boolean):
                return node.this
            if isinstance(node, exp.Neg):
                return -resolve(node.this)
            for node_type, operation in _CALCULATED_BINARY_OPERATORS.items():
                if isinstance(node, node_type):
                    return operation(resolve(node.this), resolve(node.expression))
            if isinstance(node, exp.Coalesce):
                return ibis.coalesce(
                    *(resolve(item) for item in (node.this, *node.expressions))
                )
            if isinstance(node, exp.Case):
                default = node.args.get("default")
                result = ibis.null() if default is None else resolve(default)
                for branch in reversed(node.args["ifs"]):
                    result = resolve(branch.this).ifelse(
                        resolve(branch.args["true"]), result
                    )
                return result
            raise ConversionError(
                f"calculated metric {metric.name!r} contains unsupported "
                f"expression {node.key!r}"
            )

        return resolve(parsed)

    return CalcMeasure(
        expr=calculated,
        description=metric.description,
        depends_on=dependencies,
        prefer_known=dependencies,
    )


def convert_ossie_to_bsl(
    ossie_yaml: str,
    *,
    source_model: SemanticModel,
    measure_columns: Iterable[str] = (),
) -> SemanticModel:
    """Chain an Ossie rule document onto an existing lazy semantic model.

    Nothing executes until the model is queried. ``measure_columns`` declares
    materialized evidence columns available to aggregate rule ``MEASURE`` reads.
    """
    if not isinstance(source_model, SemanticModel):
        raise ConversionError("source_model must be a SemanticModel")

    document = OSIDocument.model_validate(yaml.safe_load(ossie_yaml))
    model = _single_model(document)
    if model.relationships:
        raise ConversionError("relationships cannot be chained onto source_model")
    dataset = _single_dataset(model)
    all_fields = list(dataset.fields)
    applies_when_columns = set(_applies_when_columns(dataset).values())
    field_names = {field.name for field in all_fields}
    internal_collisions = sorted(applies_when_columns & field_names)
    if internal_collisions:
        raise ConversionError(
            "fields use reserved rule_applies_when column names: "
            + ", ".join(internal_collisions)
        )

    upstream_dimensions: dict[str, Dimension] = {}
    metrics = list(model.metrics or [])
    metric_dimensions: dict[str, Dimension] = {}
    status_metrics = [
        metric for metric in metrics if _rules_engine_payload(metric).get("semantic_role") == "rule_status"
    ]
    if dataset.source.lstrip().upper().startswith("SELECT"):
        source_model = query_rule_source_model(
            source_model,
            source=dataset.source,
            primary_key=_anchor_source_columns(dataset),
            measure_columns=measure_columns,
        )
    if status_metrics:
        status_fields = sorted(
            field.name
            for field in dataset.fields
            if _rules_engine_payload(field).get("semantic_role") == "rule_status"
        )
        if status_fields:
            raise ConversionError(
                "a rule family cannot mix row-grain status fields with measure-authored "
                "statuses: " + ", ".join(status_fields)
            )
        source_model = _aggregate_status_model(source_model, dataset=dataset, metrics=metrics)
        metric_dimensions = {
            metric.name: Dimension(
                expr=_reference(metric.name),
                description=metric.description,
                metadata=(
                    {NAMESPACE: {"field_extensions": _extensions(metric)}}
                    if _extensions(metric)
                    else {}
                ),
            )
            for metric in metrics
        }
        metrics = []
    base = _chain_base(source_model).alias(CHAIN_ALIAS)
    upstream_dimensions = {
        name: dimension
        for name, dimension in source_model.get_dimensions().items()
        if name not in metric_dimensions
    }
    upstream_names = set(source_model.table.columns) | set(upstream_dimensions)
    reused_fields = {
        field.name
        for field in dataset.fields
        if field.name in upstream_names and _source_passthrough_field(field)
    }
    collisions = sorted(
        field.name
        for field in dataset.fields
        if field.name in upstream_names and field.name not in reused_fields
    )
    if collisions:
        raise ConversionError(
            "fields would shadow upstream columns: " + ", ".join(collisions)
        )
    internal_collisions = sorted(applies_when_columns & upstream_names)
    if internal_collisions:
        raise ConversionError(
            "rule_applies_when columns would shadow upstream columns: "
            + ", ".join(internal_collisions)
        )
    sql = _projection_sql(
        dataset,
        alias=CHAIN_ALIAS,
        fields=[field for field in dataset.fields if field.name not in reused_fields],
        metrics=metrics,
    )
    projected = base.sql(sql, dialect=SQL_DIALECT)

    primary_key = set(dataset.primary_key or [])
    model_extensions = _extensions(model)
    dimensions = {}
    for field in all_fields:
        if field.name in upstream_dimensions and _source_passthrough_field(field):
            continue
        metadata: dict[str, Any] = {}
        field_extensions = _extensions(field)
        if field_extensions:
            metadata["field_extensions"] = field_extensions
        if field.name in primary_key and model_extensions:
            metadata["model_extensions"] = model_extensions

        dimensions[field.name] = Dimension(
            expr=_reference(field.name),
            description=field.description,
            is_entity=field.name in primary_key,
            metadata={NAMESPACE: metadata} if metadata else {},
        )

    semantic_model = to_semantic_table(
        projected, name=model.name, description=model.description
    ).with_dimensions(**upstream_dimensions, **metric_dimensions, **dimensions)
    if metrics:
        base_metrics = {
            metric.name: base
            for metric in metrics
            if (base := _base_metric(metric)) is not None
        }
        semantic_model = semantic_model.with_measures(
            **{
                metric.name: _measure(metric, method=base_metrics[metric.name][0])
                for metric in metrics
                if metric.name in base_metrics
            }
        )
        calculated = {
            metric.name: _calculated_measure(
                metric, measure_names=set(base_metrics)
            )
            for metric in metrics
            if metric.name not in base_metrics
        }
        if calculated:
            semantic_model = SemanticModel(
                table=semantic_model.table,
                dimensions=semantic_model.get_dimensions(),
                measures=semantic_model.get_measures(),
                calc_measures={
                    **semantic_model.get_calculated_measures(),
                    **calculated,
                },
                name=semantic_model.name,
                description=semantic_model.description,
                _source_join=semantic_model.op()._source_join,
            )
    return semantic_model
