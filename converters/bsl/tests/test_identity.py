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

"""``with_node_identity`` stamps node identity dimensions keyed on
``evaluation_key`` -- the retired ``scope_key`` name must never be accepted
as a parameter or emitted as a dimension.
"""

import ibis
import pytest
from boring_semantic_layer.ops import Dimension

from ossie_bsl import convert_ossie_to_bsl
from ossie_bsl.identity import with_node_identity

from _util import evidence_ossie

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

ROW = (
    "plan-1", "MOSAIQ", "BRAIN PLAN", "APPROVED", True, "2026-01-01", "physicist",
    "CURATIVE", "6000", "100", "30", "6000", "98.5", "HFS", "BRAIN", "LEFT",
    "IMRT", False,
)


@pytest.fixture
def con():
    connection = ibis.duckdb.connect()
    values = "(" + ", ".join("NULL" if v is None else repr(v) for v in ROW) + ")"
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


def test_with_node_identity_rejects_scope_key_keyword(evidence):
    with pytest.raises(TypeError):
        with_node_identity(evidence, scope_key="eval-1")  # noqa: E501


def test_with_node_identity_stamps_evaluation_key_not_scope_key(evidence):
    identified = with_node_identity(evidence, evaluation_key="eval-1")

    dimensions = identified.get_dimensions()
    assert "evaluation_key" in dimensions
    assert "scope_key" not in dimensions

    rows = (
        identified.query(dimensions=["evaluation_key", "evidence_node_id"])
        .execute()
        .to_dict("records")
    )
    assert rows[0]["evaluation_key"] == "eval-1"
    assert rows[0]["evidence_node_id"].startswith("evidence::eval-1::self.plan_evidence::1.2.0::")


def test_with_node_identity_rejects_conflicting_literal_and_upstream_evaluation_key(evidence):
    identified = with_node_identity(evidence, evaluation_key="eval-1")

    with pytest.raises(ValueError, match="evaluation_key conflict"):
        with_node_identity(identified, evaluation_key="eval-2")


def test_with_node_identity_reuses_upstream_evaluation_key_when_none_given(evidence):
    # Simulate an upstream source that already carries its own evaluation_key
    # dimension (e.g. a joined-in model), distinct from calling
    # with_node_identity twice on the same anchor.
    upstream = evidence.with_dimensions(
        evaluation_key=Dimension(expr=lambda table: ibis.literal("eval-1"), is_entity=False)
    )
    identified = with_node_identity(upstream, evaluation_key=None)

    rows = identified.query(dimensions=["evaluation_key"]).execute().to_dict("records")
    assert rows[0]["evaluation_key"] == "eval-1"


def test_with_node_identity_requires_a_non_empty_evaluation_key_without_upstream(evidence):
    with pytest.raises(ValueError, match="evaluation_key must be"):
        with_node_identity(evidence, evaluation_key=None)
