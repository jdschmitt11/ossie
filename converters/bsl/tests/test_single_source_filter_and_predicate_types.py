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

"""Two production behaviours the single-source path must match.

1. An evidence artifact's ``filter:`` applies on the single-table path, not only
   on the joined one. self_activity_evidence filters out one department; the
   rules engine evaluates only the surviving rows, so Ossie must too.
2. A rule predicate that is not boolean-typed (an integer, or a CASE whose every
   branch is NULL) is coerced the way DuckDB coerces ``(predicate) IS TRUE``:
   non-zero is PASS, zero is FAIL, NULL is PENDING.
"""

import ibis
import pytest

from ossie_bsl import convert_ossie_to_bsl, evaluate_rules
from ossie_rules_engine import convert_evidence_to_ossie, convert_rules_to_ossie

_EVIDENCE = """\
version: '1.1'
evidence_version: '1.0.0'
source: gold_activity
filter: source.department_name <> 'Excluded Dept'
dimensions:
  - name: activity_id
    expr: source.sk_activity
    tags: [rules_engine:anchor, node_type:activity, evidence_version:1.0.0]
  - name: department_name
    expr: source.department_name
  - name: units
    expr: source.units
"""

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
    evidence = convert_ossie_to_bsl(
        convert_evidence_to_ossie(_EVIDENCE, model_name="activity_evidence", evidence_id="self.activity"),
        tables={"gold_activity": con.table("gold_activity")},
    )
    rules = evaluate_rules(
        convert_ossie_to_bsl(convert_rules_to_ossie(_RULES, model_name="activity"), source_model=evidence)
    )
    return rules.query(dimensions=["activity_id", "has_units", "never_decided"]).execute().set_index("activity_id")


def test_the_evidence_filter_applies_on_the_single_table_path(con):
    statuses = _statuses(con)

    assert sorted(statuses.index) == [1, 2, 3]


def test_an_integer_predicate_is_coerced_like_is_true(con):
    statuses = _statuses(con)["has_units"].to_dict()

    assert statuses == {1: "PASS", 2: "FAIL", 3: "PENDING"}


def test_a_predicate_with_no_boolean_branch_is_pending_everywhere(con):
    statuses = _statuses(con)["never_decided"].to_dict()

    assert statuses == {1: "PENDING", 2: "PENDING", 3: "PENDING"}
