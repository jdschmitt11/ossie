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

"""An authored rule source may aggregate evidence itself.

dc_laterality declares its rule source as a SELECT that mixes raw aggregates
(``MAX(_data_source)``, a ``CASE`` over ``COUNT(DISTINCT ...)``) with
``MEASURE(name)`` calls, grouped by the anchor. Databricks evaluates that
query over the evidence metric view; the converter must evaluate the same
shape over the BSL evidence model, one row per group.
"""

import ibis
import pytest
from boring_semantic_layer import to_semantic_table
from boring_semantic_layer.ops import Dimension

from ossie_bsl.internals import (
    ConversionError,
    query_rule_source_model,
    rule_source_measure_names,
)


@pytest.fixture
def evidence():
    con = ibis.duckdb.connect()
    con.create_table(
        "laterality",
        ibis.memtable(
            {
                "sk_plan": ["p1", "p1", "p2", "p2", "p3"],
                "evaluation_key": ["k", "k", "k", "k", "k"],
                "dx_laterality": ["Left", "Left", "Left", "Right", None],
                "eclipse_laterality": ["Left", "Left", "Right", "Right", "Left"],
                "iso_x_mm": [-10.0, -12.0, 8.0, 9.0, -4.0],
            }
        ),
    )
    return (
        to_semantic_table(con.table("laterality"))
        .with_dimensions(
            sk_plan=Dimension(expr=lambda table: table.sk_plan, is_entity=True),
            evaluation_key=Dimension(expr=lambda table: table.evaluation_key, is_entity=True),
            dx_laterality=lambda table: table.dx_laterality,
            eclipse_laterality=lambda table: table.eclipse_laterality,
        )
        .with_measures(min_iso_x_mm=lambda table: table.iso_x_mm.min())
    )


_SOURCE = """
    SELECT
      sk_plan,
      CASE
        WHEN COUNT(DISTINCT dx_laterality) = 1 THEN MAX(dx_laterality)
        WHEN COUNT(DISTINCT dx_laterality) > 1 THEN 'Ambiguous'
        ELSE 'None'
      END AS dx_laterality,
      MEASURE(min_iso_x_mm) AS min_iso_x_mm
    FROM radonc.evidence.laterality_context
    GROUP BY sk_plan
"""


def _rows(model, columns):
    return (
        model.query(dimensions=columns)
        .execute()
        .sort_values("sk_plan")
        .reset_index(drop=True)
        .to_dict("records")
    )


def test_raw_aggregates_and_measures_evaluate_once_per_group(evidence):
    model = query_rule_source_model(evidence, source=_SOURCE, primary_key=["sk_plan"])

    assert _rows(model, ["sk_plan", "dx_laterality", "min_iso_x_mm"]) == [
        {"sk_plan": "p1", "dx_laterality": "Left", "min_iso_x_mm": -12.0},
        {"sk_plan": "p2", "dx_laterality": "Ambiguous", "min_iso_x_mm": 8.0},
        {"sk_plan": "p3", "dx_laterality": "None", "min_iso_x_mm": -4.0},
    ]


def test_structural_entity_dimensions_ride_along_with_the_group(evidence):
    model = query_rule_source_model(evidence, source=_SOURCE, primary_key=["sk_plan"])

    assert model.get_dimensions()["sk_plan"].is_entity
    assert model.get_dimensions()["evaluation_key"].is_entity
    assert _rows(model, ["sk_plan", "evaluation_key"]) == [
        {"sk_plan": "p1", "evaluation_key": "k"},
        {"sk_plan": "p2", "evaluation_key": "k"},
        {"sk_plan": "p3", "evaluation_key": "k"},
    ]


def test_measure_of_a_declared_constant_column_takes_its_value(evidence):
    """A materialised evidence relation carries former measures as columns.

    The caller vouches that the column is constant within each group; the
    converter then reads it with MAX rather than rejecting the reference.
    """
    source = """
        SELECT sk_plan, MEASURE(eclipse_laterality) AS eclipse_laterality
        FROM radonc.evidence.laterality_context
        GROUP BY sk_plan
    """
    model = query_rule_source_model(
        evidence,
        source=source,
        primary_key=["sk_plan"],
        measure_columns=["eclipse_laterality"],
    )

    assert _rows(model, ["sk_plan", "eclipse_laterality"]) == [
        {"sk_plan": "p1", "eclipse_laterality": "Left"},
        {"sk_plan": "p2", "eclipse_laterality": "Right"},
        {"sk_plan": "p3", "eclipse_laterality": "Left"},
    ]


def test_measure_of_an_undeclared_column_is_rejected(evidence):
    source = """
        SELECT sk_plan, MEASURE(eclipse_laterality) AS eclipse_laterality
        FROM radonc.evidence.laterality_context
        GROUP BY sk_plan
    """
    with pytest.raises(ConversionError, match="MEASURE\\(eclipse_laterality\\)"):
        query_rule_source_model(evidence, source=source, primary_key=["sk_plan"])


def test_an_aggregate_over_an_unknown_column_is_rejected(evidence):
    source = """
        SELECT sk_plan, MAX(not_a_column) AS last_value
        FROM radonc.evidence.laterality_context
        GROUP BY sk_plan
    """
    with pytest.raises(ConversionError, match="not_a_column"):
        query_rule_source_model(evidence, source=source, primary_key=["sk_plan"])


def test_a_non_aggregate_select_item_is_rejected(evidence):
    source = """
        SELECT sk_plan, dx_laterality AS first_seen
        FROM radonc.evidence.laterality_context
        GROUP BY sk_plan
    """
    with pytest.raises(ConversionError, match="aggregate"):
        query_rule_source_model(evidence, source=source, primary_key=["sk_plan"])


def test_measure_names_are_reported_for_the_caller(evidence):
    assert rule_source_measure_names(_SOURCE) == ["min_iso_x_mm"]


def test_the_primary_key_may_be_a_structural_entity_dimension(evidence):
    """The rules engine anchors on subject_node_id while the authored query
    groups by the natural key; an entity dimension riding along with the
    group satisfies the anchor."""
    source = """
        SELECT sk_plan, MAX(dx_laterality) AS dx_laterality
        FROM radonc.evidence.laterality_context
        GROUP BY sk_plan
    """
    model = query_rule_source_model(evidence, source=source, primary_key=["evaluation_key"])

    assert _rows(model, ["sk_plan", "evaluation_key"])[0] == {"sk_plan": "p1", "evaluation_key": "k"}


def test_a_primary_key_absent_from_the_group_is_rejected(evidence):
    source = """
        SELECT sk_plan, MAX(dx_laterality) AS dx_laterality
        FROM radonc.evidence.laterality_context
        GROUP BY sk_plan
    """
    with pytest.raises(ConversionError, match="primary key"):
        query_rule_source_model(evidence, source=source, primary_key=["dx_laterality"])
