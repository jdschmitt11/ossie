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
import re
from typing import Any, Mapping

import ibis
import yaml
from boring_semantic_layer import SemanticModel, to_semantic_table
from boring_semantic_layer.ops import CalcMeasure, Dimension, Measure
from ossie import (
    OSIDataset,
    OSIDialect,
    OSIDocument,
    OSIField,
    OSIRelationship,
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
_DATABRICKS_VENDOR = "DATABRICKS"

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


def _raw_column_name(dataset_index: int, column: str) -> str:
    return f"__ossie_raw__{dataset_index}__{column}"


def _raw_projection_sql(table: Any, *, alias: str, dataset_index: int) -> str:
    columns = ", ".join(
        f"{column} AS {_raw_column_name(dataset_index, column)}" for column in table.columns
    )
    return f"SELECT {columns} FROM {alias}"


def _joined_expression(
    expression: str, *, dataset_name: str, alias_indexes: dict[str, int]
) -> str:
    """Bind an OSSIE expression to the raw columns retained by the Ibis joins."""
    expression = _bind_deep_join_paths(expression, alias_indexes=alias_indexes)
    try:
        parsed = parse_one(expression, read=SQL_DIALECT)
    except ParseError as exc:
        raise ConversionError(f"cannot parse joined expression {expression!r}") from exc

    def bind_column(node: exp.Expression) -> exp.Expression:
        if not isinstance(node, exp.Column) or len(node.parts) < 2:
            return node
        qualifier = ".".join(part.name for part in node.parts[:-1])
        target = alias_indexes.get(qualifier)
        if target is None:
            return node
        return exp.column(_raw_column_name(target, node.name))

    authored_as_bare_column = isinstance(parsed, exp.Column) and len(parsed.parts) == 1
    rewritten = parsed.transform(bind_column)
    if authored_as_bare_column:
        if parsed.name.startswith("__ossie_raw__"):
            return parsed.name
        return _raw_column_name(alias_indexes[dataset_name], parsed.name)
    return rewritten.sql(dialect=SQL_DIALECT)


def _bind_deep_join_paths(expression: str, *, alias_indexes: Mapping[str, int]) -> str:
    """Bind declared nested join paths before SQLGlot parses them.

    Databricks metric views allow arbitrarily deep relationship paths, while a
    SQL column has at most three qualifiers in SQLGlot's AST. Replace only the
    declared path-plus-column tokens outside quoted literals; ordinary SQL is
    still parsed and rewritten by :func:`_joined_expression` below.
    """
    nested_paths = {
        path: index for path, index in alias_indexes.items() if "." in path
    }
    if not nested_paths:
        return expression

    path_pattern = "|".join(
        re.escape(path) for path in sorted(nested_paths, key=len, reverse=True)
    )
    reference = re.compile(
        rf"(?<![A-Za-z0-9_])(?P<path>{path_pattern})\.(?P<column>[A-Za-z_][A-Za-z0-9_]*)"
    )
    quoted = re.compile(r"('(?:''|[^'])*'|\"(?:\"\"|[^\"])*\")")

    def replace(segment: str) -> str:
        return reference.sub(
            lambda match: _raw_column_name(
                nested_paths[match.group("path")], match.group("column")
            ),
            segment,
        )

    return "".join(
        part if quoted.fullmatch(part) else replace(part)
        for part in quoted.split(expression)
    )


def _databricks_filter(model: OSISemanticModel) -> str | None:
    payload = _extensions(model).get(_DATABRICKS_VENDOR)
    if payload is None:
        return None
    if not isinstance(payload, dict) or "filter" not in payload:
        return None
    filter_sql = payload["filter"]
    if not isinstance(filter_sql, str) or not filter_sql.strip():
        raise ConversionError("DATABRICKS filter must be a non-empty SQL string")
    return filter_sql


def _joined_projection(
    model: OSISemanticModel, tables: Mapping[str, Any]
) -> tuple[OSIDataset, Any, dict[str, int]]:
    """Build the declared many-to-one relationship tree with Ibis left joins."""
    datasets = model.datasets or []
    if not datasets:
        raise ConversionError("semantic model declares no datasets")
    root = datasets[0]
    by_name = {dataset.name: dataset for dataset in datasets}
    if len(by_name) != len(datasets):
        raise ConversionError("semantic model declares duplicate dataset names")

    for relationship in model.relationships or []:
        if relationship.from_dataset not in by_name or relationship.to not in by_name:
            raise ConversionError(f"relationship {relationship.name!r} references an unknown dataset")
        if len(relationship.from_columns) != len(relationship.to_columns):
            raise ConversionError(f"relationship {relationship.name!r} has mismatched join columns")
    projections = {}
    dataset_indexes = {dataset.name: index for index, dataset in enumerate(datasets)}
    for dataset in datasets:
        if dataset.source not in tables:
            raise ConversionError(f"no table bound for source {dataset.source!r}")
        input_alias = f"__ossie_input__{dataset_indexes[dataset.name]}"
        table = tables[dataset.source].alias(input_alias)
        projections[dataset.name] = table.sql(
            _raw_projection_sql(table, alias=input_alias, dataset_index=dataset_indexes[dataset.name]),
            dialect=SQL_DIALECT,
        ).alias(f"__ossie_dataset__{dataset.name}")

    joined = projections[root.name]
    joined_names = {root.name}
    alias_indexes = {"source": dataset_indexes[root.name], root.name: dataset_indexes[root.name]}
    dataset_paths = {root.name: ""}
    pending = list(model.relationships or [])
    while pending:
        for relationship in pending[:]:
            if relationship.from_dataset not in joined_names:
                continue
            if relationship.to in joined_names:
                raise ConversionError(
                    f"relationship {relationship.name!r} does not form a tree from {root.name!r}"
                )
            child = projections[relationship.to]
            join_filter = _relationship_join_filter(relationship)
            if join_filter is not None:
                child_alias = f"__ossie_filtered__{relationship.to}"
                filter_sql = _joined_expression(
                    join_filter,
                    dataset_name=relationship.to,
                    alias_indexes={
                        relationship.to: dataset_indexes[relationship.to],
                    },
                )
                child = child.alias(child_alias).sql(
                    f"SELECT * FROM {child_alias} WHERE {filter_sql}",
                    dialect=SQL_DIALECT,
                )
            predicates = [
                joined[_raw_column_name(dataset_indexes[relationship.from_dataset], from_column)]
                == child[_raw_column_name(dataset_indexes[relationship.to], to_column)]
                for from_column, to_column in zip(
                    relationship.from_columns, relationship.to_columns, strict=True
                )
            ]
            joined = joined.left_join(child, predicates)
            joined_names.add(relationship.to)
            parent_path = dataset_paths[relationship.from_dataset]
            path = f"{parent_path}.{relationship.to}".lstrip(".")
            dataset_paths[relationship.to] = path
            alias_indexes[path] = dataset_indexes[relationship.to]
            alias_indexes.setdefault(relationship.to, dataset_indexes[relationship.to])
            pending.remove(relationship)
            break
        else:
            unresolved = ", ".join(relationship.name or "<unnamed>" for relationship in pending)
            raise ConversionError(
                f"relationships must form a directed tree from {root.name!r}: {unresolved}"
            )

    fields = [field for dataset in datasets for field in (dataset.fields or [])]
    field_names = [field.name for field in fields]
    if len(set(field_names)) != len(field_names):
        raise ConversionError("joined datasets declare duplicate field names")
    joined_alias = "__ossie_joined"
    selections = [
        f"{_joined_expression(_pick_expression(field), dataset_name=dataset.name, alias_indexes=alias_indexes)} AS {field.name}"
        for dataset in datasets
        for field in (dataset.fields or [])
    ]
    selections.extend(
        f"{_metric_argument_expression(metric, dataset=root, alias_indexes=alias_indexes)} "
        f"AS {_metric_argument_name(metric.name)}"
        for metric in (model.metrics or [])
        if _base_metric(metric) is not None
    )
    filter_sql = _databricks_filter(model)
    where = (
        ""
        if filter_sql is None
        else " WHERE " + _joined_expression(
            filter_sql, dataset_name=root.name, alias_indexes=alias_indexes
        )
    )
    projected = joined.alias(joined_alias).sql(
        f"SELECT {', '.join(selections)} FROM {joined_alias}{where}", dialect=SQL_DIALECT
    )
    return root, projected, alias_indexes


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
    obj: OSIDataset | OSIField | OSIRelationship | OSISemanticModel,
) -> dict[str, Any]:
    return {
        extension.vendor_name: json.loads(extension.data)
        for extension in (obj.custom_extensions or [])
    }


def _relationship_join_filter(relationship: OSIRelationship) -> str | None:
    payload = _extensions(relationship).get(_RULES_ENGINE_VENDOR)
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ConversionError(
            f"relationship {relationship.name!r}: RULES_ENGINE metadata must be an object"
        )
    join_filter = payload.get("join_filter")
    if join_filter is None:
        return None
    if not isinstance(join_filter, str) or not join_filter.strip():
        raise ConversionError(
            f"relationship {relationship.name!r}: join_filter must be a non-empty SQL string"
        )
    return join_filter


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
    passthrough: bool,
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
        f"{_metric_argument_expression(metric, dataset=dataset, alias_indexes=None)} "
        f"AS {_metric_argument_name(metric.name)}"
        for metric in (metrics or [])
        if _base_metric(metric) is not None
    )
    if passthrough:
        selections.insert(0, f"{SOURCE_ALIAS}.*")
    columns = ", ".join(selections)
    if passthrough:
        return f"SELECT {columns} FROM {alias} AS {SOURCE_ALIAS}"
    return f"SELECT {columns} FROM {SOURCE_ALIAS}"


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


def _measure_source_spec(source: str) -> tuple[list[str], list[str]]:
    """Validate the explicit metric-view query shape used as a rule source."""
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
    for item in query.expressions:
        if isinstance(item, exp.Column) and not item.table:
            dimensions.append(item.name)
            continue
        if not isinstance(item, exp.Alias) or not isinstance(item.this, exp.Anonymous):
            raise ConversionError(
                "rule measure source SELECT items must be bare dimensions or MEASURE(name) AS name"
            )
        measure_call = item.this
        if measure_call.name.upper() != "MEASURE" or len(measure_call.expressions) != 1:
            raise ConversionError("rule measure source supports only MEASURE(name) AS name")
        measure_column = measure_call.expressions[0]
        if not isinstance(measure_column, exp.Column) or measure_column.table:
            raise ConversionError("MEASURE() must name one unqualified evidence measure")
        if item.alias != measure_column.name:
            raise ConversionError("MEASURE(name) must be aliased to the same name")
        measures.append(measure_column.name)
    if not dimensions or not measures or len(set(dimensions)) != len(dimensions):
        raise ConversionError(
            "rule measure source requires unique dimensions and at least one MEASURE(name)"
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
    return dimensions, measures


def query_rule_source_model(
    source_model: SemanticModel,
    *,
    source: str,
    primary_key: list[str] | None = None,
) -> SemanticModel:
    """Evaluate a declared metric-view query lazily through BSL."""
    dimensions, measures = _measure_source_spec(source)
    upstream_dimensions = source_model.get_dimensions()
    missing = sorted(set(dimensions).difference(upstream_dimensions))
    if missing:
        raise ConversionError(
            "rule measure source references unknown evidence dimensions: " + ", ".join(missing)
        )
    missing_anchor = sorted(set(primary_key or []).difference(dimensions))
    if missing_anchor:
        raise ConversionError(
            "rule measure source must select its primary key dimensions: "
            + ", ".join(missing_anchor)
        )
    structural_dimensions = [
        name
        for name, dimension in upstream_dimensions.items()
        if dimension.is_entity and name not in dimensions
    ]
    aggregate = source_model.query(
        dimensions=[*dimensions, *structural_dimensions], measures=measures
    )
    return _rebind_semantic_relation(aggregate.to_untagged(), source_model=source_model)


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
    alias_indexes: Mapping[str, int] | None,
) -> str:
    base = _base_metric(metric)
    if base is None:
        raise ConversionError(f"metric {metric.name!r} is not a base aggregate")
    argument = base[1]
    if alias_indexes is not None:
        return _joined_expression(
            argument.sql(dialect=SQL_DIALECT),
            dataset_name=dataset.name,
            alias_indexes=dict(alias_indexes),
        )

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
    tables: Mapping[str, Any] | None = None,
    source_model: SemanticModel | None = None,
) -> SemanticModel:
    """Build a lazy BSL ``SemanticModel`` from an Ossie YAML document.

    Pass ``tables`` to project over a physical table, or ``source_model`` to
    chain onto an existing model. Nothing is executed; the result materialises
    only on ``.query(...).execute()``.
    """
    if (tables is None) == (source_model is None):
        raise ConversionError("pass exactly one of tables= or source_model=")

    document = OSIDocument.model_validate(yaml.safe_load(ossie_yaml))
    model = _single_model(document)
    if model.relationships:
        if source_model is not None:
            raise ConversionError("relationships cannot be chained onto source_model")
        dataset, projected, metric_aliases = _joined_projection(model, tables)
        all_fields = [field for item in model.datasets or [] for field in (item.fields or [])]
    else:
        dataset = _single_dataset(model)
        metric_aliases = None
        all_fields = list(dataset.fields)
        projected = None
    applies_when_columns = set(_applies_when_columns(dataset).values())
    field_names = {field.name for field in all_fields}
    internal_collisions = sorted(applies_when_columns & field_names)
    if internal_collisions:
        raise ConversionError(
            "fields use reserved rule_applies_when column names: "
            + ", ".join(internal_collisions)
        )

    upstream_dimensions: dict[str, Dimension] = {}
    if model.relationships:
        sql = None
    elif source_model is not None:
        if dataset.source.lstrip().upper().startswith("SELECT"):
            source_model = query_rule_source_model(
                source_model, source=dataset.source, primary_key=list(dataset.primary_key or [])
            )
        base = source_model.table.alias(CHAIN_ALIAS)
        upstream_dimensions = source_model.get_dimensions()
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
            passthrough=True,
            fields=[field for field in dataset.fields if field.name not in reused_fields],
            metrics=list(model.metrics or []),
        )
    else:
        if dataset.source not in tables:
            raise ConversionError(f"no table bound for source {dataset.source!r}")
        base = tables[dataset.source].alias(SOURCE_ALIAS)
        sql = _projection_sql(
            dataset,
            alias=SOURCE_ALIAS,
            passthrough=False,
            metrics=list(model.metrics or []),
        )

    if not model.relationships:
        projected = base.sql(sql, dialect=SQL_DIALECT)
    metrics = list(model.metrics or [])

    primary_key = set(dataset.primary_key or [])
    model_extensions = _extensions(model)
    dimensions = {}
    for field in all_fields:
        if source_model is not None and field.name in upstream_dimensions and _source_passthrough_field(field):
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
    ).with_dimensions(**upstream_dimensions, **dimensions)
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
