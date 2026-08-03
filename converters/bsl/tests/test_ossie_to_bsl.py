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

import ibis
import pandas as pd
import pytest
from boring_semantic_layer import SemanticModel
from boring_semantic_layer.ops import Dimension

from ossie_bsl import convert_ossie_to_bsl
from ossie_bsl.internals import ConversionError

from _util import evidence_ossie, rules_ossie

# Every ``source.<column>`` the self-plan evidence artifact references.
COLUMNS = (
    "sk_plan",
    "_data_source",
    "plan_name",
    "plan_status",
    "plan_approved",
    "plan_approved_date_time",
    "plan_approved_user",
    "plan_intent",
    "prescribed_dose",
    "prescribed_percentage",
    "number_of_fractions",
    "total_dose",
    "coverage_percent",
    "treatment_orientation",
    "tx_site",
    "laterality",
    "treatment_type",
    "is_replan",
)

# One row per rule status the pilot must produce.
ROWS = [
    (
        "plan-pass", "MOSAIQ", "BRAIN PLAN", "APPROVED", True, "2026-01-01", "physicist",
        "CURATIVE", "6000", "100", "30", "6000", "98.5", "HFS", "BRAIN", "LEFT",
        "IMRT", False,
    ),
    (
        "plan-fail", "MOSAIQ", "", "DRAFT", False, None, None,
        "", "0", "0", "0", "0", "0", "", "", "", "", False,
    ),
    (
        "plan-pending", None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
    ),
]

ROW_COUNT = len(ROWS)


@pytest.fixture
def con():
    connection = ibis.duckdb.connect()
    values = ", ".join(
        "(" + ", ".join("NULL" if v is None else repr(v) for v in row) + ")" for row in ROWS
    )
    connection.raw_sql(
        f"CREATE TABLE gold_plan_with_sk AS "
        f"SELECT * FROM (VALUES {values}) t({', '.join(COLUMNS)})"
    )
    return connection


@pytest.fixture
def evidence(con):
    return convert_ossie_to_bsl(
        evidence_ossie(), tables={"gold_plan_with_sk": con.table("gold_plan_with_sk")}
    )


def test_evidence_becomes_a_semantic_model(evidence):
    assert isinstance(evidence, SemanticModel)
    assert evidence.name == "self_plan_evidence"
    assert evidence.description.startswith("display_name: Plan Evidence")


def test_authored_sql_is_transpiled_and_evaluated(evidence):
    rows = (
        evidence.query(dimensions=["sk_plan", "total_dose", "dose_per_fraction_cgy"])
        .execute()
        .set_index("sk_plan")
        .to_dict("index")
    )

    assert rows["plan-pass"] == {"total_dose": 6000.0, "dose_per_fraction_cgy": 200.0}
    assert rows["plan-fail"]["total_dose"] == 0.0
    # NULL from the CASE ELSE branch; pandas renders a null float as NaN.
    assert pd.isna(rows["plan-fail"]["dose_per_fraction_cgy"])
    assert pd.isna(rows["plan-pending"]["total_dose"])


def test_the_primary_key_becomes_an_entity_dimension(evidence):
    dimensions = evidence.get_dimensions()

    assert dimensions["sk_plan"].is_entity is True
    assert dimensions["total_dose"].is_entity is False


def test_field_extensions_are_namespaced_in_dimension_metadata(evidence):
    metadata = evidence.get_dimensions()["sk_plan"].metadata

    assert metadata["ossie"]["field_extensions"] == {
        "RULES_ENGINE": {
            "_v": 1,
            "anchor": True,
            "node_type": "plan",
            "evidence_version": "1.2.0",
        }
    }
    assert evidence.get_dimensions()["total_dose"].metadata == {}


def test_model_extensions_are_mirrored_onto_the_primary_key_dimension(evidence):
    metadata = evidence.get_dimensions()["sk_plan"].metadata

    assert metadata["ossie"]["model_extensions"]["RULES_ENGINE"]["evidence_id"] == (
        "self.plan_evidence"
    )


def test_rules_chain_onto_the_evidence_model(evidence):
    predicates = convert_ossie_to_bsl(rules_ossie(), source_model=evidence)

    rows = (
        predicates.query(dimensions=["plan_id", "total_dose_present"])
        .execute()
        .set_index("plan_id")
        .to_dict("index")
    )
    assert rows["plan-pass"]["total_dose_present"] is True
    assert rows["plan-fail"]["total_dose_present"] is False
    assert rows["plan-pending"]["total_dose_present"] is None


def test_chaining_does_not_produce_a_duplicate_source_cte(evidence):
    """Regression: both stages author expressions qualified ``source.``."""
    predicates = convert_ossie_to_bsl(rules_ossie(), source_model=evidence)

    assert predicates.query(dimensions=["plan_id"]).execute().shape[0] == ROW_COUNT


def test_rule_metadata_survives_the_chain(evidence):
    predicates = convert_ossie_to_bsl(rules_ossie(), source_model=evidence)

    extensions = predicates.get_dimensions()["total_dose_present"].metadata["ossie"][
        "field_extensions"
    ]
    assert extensions["RULES_ENGINE"]["rule_severity"] == "HIGH"
    assert extensions["RULES_ENGINE"]["rule_subject_columns"] == ["total_dose"]


def test_construction_materialises_no_rows(evidence):
    predicates = convert_ossie_to_bsl(rules_ossie(), source_model=evidence)

    assert isinstance(predicates.table, ibis.expr.types.Table)
    assert predicates.query(dimensions=["plan_id"]).sql() is not None


def test_a_semantic_only_upstream_dimension_survives_the_chain(evidence):
    """Regression: chaining used to build from ``source_model.table`` alone,

    a physical-columns-only view that drops anything BSL knows but Ibis
    cannot represent -- ``is_entity``, ``metadata``, dimensions with no
    backing column at all.
    """
    enriched = evidence.with_dimensions(
        subject_node_id=Dimension(
            expr=lambda t: t.sk_plan.cast("string"),
            is_entity=True,
            metadata={"ossie": {"synthetic": True}},
        )
    )

    predicates = convert_ossie_to_bsl(rules_ossie(), source_model=enriched)

    assert "subject_node_id" in predicates.get_dimensions()
    dimension = predicates.get_dimensions()["subject_node_id"]
    assert dimension.is_entity is True
    assert dimension.metadata == {"ossie": {"synthetic": True}}

    rows = predicates.query(dimensions=["subject_node_id"]).execute()
    assert set(rows["subject_node_id"]) == {"plan-pass", "plan-fail", "plan-pending"}


def test_every_upstream_dimension_survives_the_chain_unchanged(evidence):
    predicates = convert_ossie_to_bsl(rules_ossie(), source_model=evidence)

    upstream = evidence.get_dimensions()
    chained = predicates.get_dimensions()
    for name, dimension in upstream.items():
        assert name in chained, f"upstream dimension {name!r} did not survive chaining"
        assert chained[name].is_entity == dimension.is_entity
        assert chained[name].metadata == dimension.metadata


def test_a_rule_name_colliding_with_an_upstream_dimension_raises(evidence):
    """A dimension that carries no physical column must still be guarded."""
    enriched = evidence.with_dimensions(
        synthetic_only=Dimension(expr=lambda t: t.sk_plan.cast("string"))
    )
    colliding = rules_ossie().replace("name: total_dose_present", "name: synthetic_only")

    with pytest.raises(ConversionError, match="synthetic_only"):
        convert_ossie_to_bsl(colliding, source_model=enriched)


def test_a_field_colliding_with_an_upstream_column_raises(evidence):
    colliding = rules_ossie().replace("name: total_dose_present", "name: total_dose")

    with pytest.raises(ConversionError, match="total_dose"):
        convert_ossie_to_bsl(colliding, source_model=evidence)


def test_requiring_exactly_one_model_and_dataset(con):
    doubled = evidence_ossie().replace(
        "semantic_model:\n", "semantic_model:\n- name: extra\n  datasets: []\n"
    )

    with pytest.raises(ConversionError, match="exactly one semantic model"):
        convert_ossie_to_bsl(
            doubled, tables={"gold_plan_with_sk": con.table("gold_plan_with_sk")}
        )


def test_a_missing_table_binding_raises():
    with pytest.raises(ConversionError, match="gold_plan_with_sk"):
        convert_ossie_to_bsl(evidence_ossie(), tables={})


def test_passing_neither_tables_nor_source_model_raises():
    with pytest.raises(ConversionError, match="tables"):
        convert_ossie_to_bsl(evidence_ossie())
