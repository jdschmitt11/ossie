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

"""Integer and all-null rule predicates retain their status semantics."""

import ibis
import pytest

from ossie_bsl import convert_ossie_to_bsl, evaluate_rules
from ossie_rules_engine import convert_rules_to_ossie

from _util import native_model

_RULES = """\
version: '1.1'
rule_family_version: '1.0.0'
evidence_id: 'self.activity'
evidence_version: '1.0.0'
render_hint: presence
source: radonc.evidence.activity
use_cases: [patient]
dimensions:
  - name: activity_id
    expr: source.activity_id
    tags: [rules_engine:anchor, node_type:activity, semantic_role:entity_key]
  - name: has_units
    expr: source.units
    tags:
      - semantic_role:rule_status
      - rule_id:A-001
      - rule_severity:LOW
      - rule_family_id:activity
      - rule_subject_columns:units
      - rule_plain_english:Units must be recorded.
  - name: never_decided
    expr: CASE WHEN source.units IS NULL THEN NULL ELSE NULL END
    tags:
      - semantic_role:rule_status
      - rule_id:A-002
      - rule_severity:LOW
      - rule_family_id:activity
      - rule_subject_columns:units
      - rule_plain_english:Suspended rule; every row is pending.
"""


@pytest.fixture
def con():
    connection = ibis.duckdb.connect()
    connection.raw_sql(
        "CREATE TABLE gold_activity (sk_activity BIGINT, department_name VARCHAR, units INTEGER)"
    )
    connection.raw_sql(
        "INSERT INTO gold_activity VALUES "
        "(1, 'Main Dept', 3), (2, 'Main Dept', 0), (3, 'Main Dept', NULL), "
        "(4, 'Excluded Dept', 5)"
    )
    return connection


def _statuses(con):
    table = con.table("gold_activity").rename(activity_id="sk_activity")
    table = table.filter(table.department_name != "Excluded Dept")
    evidence = native_model(table, entities=("activity_id",))
    rules = evaluate_rules(
        convert_ossie_to_bsl(convert_rules_to_ossie(_RULES, model_name="activity"), source_model=evidence)
    )
    return rules.query(dimensions=["activity_id", "has_units", "never_decided"]).execute().set_index("activity_id")


def test_an_integer_predicate_is_coerced_like_is_true(con):
    statuses = _statuses(con)["has_units"].to_dict()

    assert statuses == {1: "PASS", 2: "FAIL", 3: "PENDING"}


def test_a_predicate_with_no_boolean_branch_is_pending_everywhere(con):
    statuses = _statuses(con)["never_decided"].to_dict()

    assert statuses == {1: "PENDING", 2: "PENDING", 3: "PENDING"}
