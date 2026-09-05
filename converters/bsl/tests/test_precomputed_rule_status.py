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

"""A precomputed rule status is projected as authored, not re-encoded.

Most rules author a boolean predicate that evaluate_rules maps to
PASS/FAIL/PENDING. Families such as dc_laterality and
ms_physics_assessment_window instead author the complete status string
themselves (``rule_status_encoding: precomputed``). Running those through the
boolean mapping is wrong twice over: the value is already a status, and
coercing a VARCHAR through a boolean branch fails outright.
"""

import ibis
import pytest
from boring_semantic_layer import SemanticModel

from ossie_bsl import convert_ossie_to_bsl, evaluate_rules
from ossie_rules_engine import convert_evidence_to_ossie, convert_rules_to_ossie

_EVIDENCE_YAML = """\
version: '1.1'
evidence_version: '1.0.0'
source: gold_laterality
dimensions:
  - name: plan_id
    expr: source.sk_plan
    tags: [rules_engine:anchor, node_type:plan, evidence_version:1.0.0]
  - name: laterality_status
    expr: source.laterality_status
"""

_RULES_YAML = """\
version: '1.1'
rule_family_version: '1.0.0'
evidence_id: 'self.laterality'
evidence_version: '1.0.0'
render_hint: presence
source: radonc.evidence.laterality
use_cases: [patient]
dimensions:
  - name: plan_id
    expr: source.plan_id
    tags: [rules_engine:anchor, node_type:plan, semantic_role:entity_key]
  - name: laterality_agrees
    expr: source.laterality_status
    tags:
      - semantic_role:rule_status
      - rule_status_encoding:precomputed
      - rule_id:DC-021
      - rule_severity:HIGH
      - rule_family_id:dc_laterality
      - rule_subject_columns:laterality_status
      - rule_plain_english:Laterality must agree.
"""


@pytest.fixture
def con():
    connection = ibis.duckdb.connect()
    connection.raw_sql("CREATE TABLE gold_laterality (sk_plan VARCHAR, laterality_status VARCHAR)")
    connection.raw_sql(
        "INSERT INTO gold_laterality VALUES "
        "('plan-pass', 'PASS'), ('plan-fail', 'FAIL'), "
        "('plan-na', 'NOT_APPLICABLE'), ('plan-null', NULL)"
    )
    return connection


def _rules(con) -> SemanticModel:
    evidence = convert_ossie_to_bsl(
        convert_evidence_to_ossie(
            _EVIDENCE_YAML, model_name="laterality_evidence", evidence_id="self.laterality"
        ),
        tables={"gold_laterality": con.table("gold_laterality")},
    )
    predicates = convert_ossie_to_bsl(
        convert_rules_to_ossie(_RULES_YAML, model_name="dc_laterality"),
        source_model=evidence,
    )
    return evaluate_rules(predicates)


def test_an_authored_status_string_is_projected_unchanged(con):
    statuses = (
        _rules(con)
        .query(dimensions=["plan_id", "laterality_agrees"])
        .execute()
        .set_index("plan_id")["laterality_agrees"]
        .to_dict()
    )

    assert statuses["plan-pass"] == "PASS"
    assert statuses["plan-fail"] == "FAIL"
    assert statuses["plan-na"] == "NOT_APPLICABLE"


def test_a_null_precomputed_status_is_projected_as_null(con):
    """The author owns a precomputed status vocabulary; a null is not re-encoded.

    This matches the rules engine's render_precomputed_status, which projects
    the authored expression unchanged."""
    import pandas as pd

    statuses = (
        _rules(con)
        .query(dimensions=["plan_id", "laterality_agrees"])
        .execute()
        .set_index("plan_id")["laterality_agrees"]
        .to_dict()
    )

    assert pd.isna(statuses["plan-null"])
