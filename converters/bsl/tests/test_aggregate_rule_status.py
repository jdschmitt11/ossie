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

"""A rule status authored as a measure is one verdict per subject.

ms_physics_assessment_window binds to document evidence that fans out (one
plan document, zero or more physics assessments) and authors its status as
an aggregate over that fan-out. The converter collapses the evidence to one
row per anchor: rule subject columns are carried with MAX, every measure is
evaluated in the same GROUP BY, and the statuses come out as dimensions that
evaluate_rules recognises exactly like row-grain ones.
"""

import ibis
import pytest
from boring_semantic_layer import SemanticModel

from ossie_bsl import convert_ossie_to_bsl, evaluate_rules
from ossie_bsl.internals import ConversionError
from ossie_rules_engine import convert_evidence_to_ossie, convert_rules_to_ossie

_EVIDENCE_YAML = """\
version: '1.1'
evidence_version: '1.0.0'
source: gold_physics
dimensions:
  - name: sk_plan_document
    expr: source.sk_plan_document
    tags: [rules_engine:anchor, node_type:plan_document, evidence_version:1.0.0]
  - name: physicist_initials
    expr: source.physicist_initials
  - name: in_window
    expr: source.in_window
"""

_RULES_HEADER = """\
version: '1.1'
rule_family_version: '1.0.0'
evidence_id: 'document.physics'
evidence_version: '1.0.0'
render_hint: presence
source: radonc.evidence.document_physics
use_cases: [patient]
dimensions:
  - name: sk_plan_document
    expr: source.sk_plan_document
    tags: [rules_engine:anchor, node_type:plan_document, semantic_role:entity_key]
"""

_PRECOMPUTED_RULES_YAML = _RULES_HEADER + """\
measures:
  - name: assessments
    expr: COUNT(source.in_window)
  - name: physics_assessment_in_window
    expr: |-
      CASE
        WHEN COUNT(source.in_window) = 0 THEN 'NOT_APPLICABLE'
        WHEN MAX(source.in_window) > 0 THEN 'PASS'
        ELSE 'FAIL'
      END
    tags:
      - semantic_role:rule_status
      - rule_status_encoding:precomputed
      - rule_evaluation_grain:subject
      - rule_id:MS-015
      - rule_severity:HIGH
      - rule_family_id:ms_physics
      - rule_subject_columns:physicist_initials,assessments
      - rule_plain_english:An assessment must land in the window.
"""

_BOOLEAN_RULES_YAML = _RULES_HEADER + """\
measures:
  - name: any_in_window
    expr: MAX(source.in_window) > 0
    tags:
      - semantic_role:rule_status
      - rule_evaluation_grain:subject
      - rule_id:MS-016
      - rule_severity:HIGH
      - rule_family_id:ms_physics
      - rule_subject_columns:physicist_initials
      - rule_plain_english:An assessment must land in the window.
"""

_MIXED_GRAIN_RULES_YAML = _RULES_HEADER + """\
  - name: initials_present
    expr: source.physicist_initials IS NOT NULL
    tags:
      - semantic_role:rule_status
      - rule_id:MS-017
      - rule_severity:LOW
      - rule_family_id:ms_physics
      - rule_subject_columns:physicist_initials
      - rule_plain_english:Initials must be present.
""" + _BOOLEAN_RULES_YAML[len(_RULES_HEADER):]


@pytest.fixture
def con():
    connection = ibis.duckdb.connect()
    connection.raw_sql(
        "CREATE TABLE gold_physics "
        "(sk_plan_document VARCHAR, physicist_initials VARCHAR, in_window INTEGER)"
    )
    connection.raw_sql(
        "INSERT INTO gold_physics VALUES "
        "('doc-pass', 'AB', 0), ('doc-pass', 'AB', 1), "
        "('doc-fail', 'CD', 0), "
        "('doc-none', NULL, NULL)"
    )
    return connection


def _evidence(con) -> SemanticModel:
    return convert_ossie_to_bsl(
        convert_evidence_to_ossie(
            _EVIDENCE_YAML, model_name="physics_evidence", evidence_id="document.physics"
        ),
        tables={"gold_physics": con.table("gold_physics")},
    )


def _rules(con, rules_yaml: str) -> SemanticModel:
    return evaluate_rules(
        convert_ossie_to_bsl(
            convert_rules_to_ossie(rules_yaml, model_name="ms_physics"),
            source_model=_evidence(con),
        )
    )


def _by_document(model, columns):
    return (
        model.query(dimensions=["sk_plan_document", *columns])
        .execute()
        .set_index("sk_plan_document")
        .to_dict("index")
    )


def test_a_precomputed_aggregate_status_is_one_verdict_per_subject(con):
    rows = _by_document(_rules(con, _PRECOMPUTED_RULES_YAML), ["physics_assessment_in_window"])

    assert rows == {
        "doc-pass": {"physics_assessment_in_window": "PASS"},
        "doc-fail": {"physics_assessment_in_window": "FAIL"},
        "doc-none": {"physics_assessment_in_window": "NOT_APPLICABLE"},
    }


def test_aggregate_status_can_chain_after_an_aggregate_rule_source(con):
    rules_yaml = _PRECOMPUTED_RULES_YAML.replace(
        "source: radonc.evidence.document_physics",
        """source: |-
  SELECT
    sk_plan_document,
    MAX(physicist_initials) AS physicist_initials,
    MAX(in_window) AS in_window
  FROM radonc.evidence.document_physics
  GROUP BY sk_plan_document""",
    )

    rows = _by_document(_rules(con, rules_yaml), ["physics_assessment_in_window"])

    assert rows == {
        "doc-pass": {"physics_assessment_in_window": "PASS"},
        "doc-fail": {"physics_assessment_in_window": "FAIL"},
        "doc-none": {"physics_assessment_in_window": "NOT_APPLICABLE"},
    }


def test_aggregate_status_preserves_its_applies_when_predicate(con):
    rules_yaml = _PRECOMPUTED_RULES_YAML.replace(
        "      - rule_id:MS-015",
        "      - rule_id:MS-015\n"
        "      - rule_applies_when:MAX(source.in_window) IS NOT NULL",
    )

    rows = _by_document(_rules(con, rules_yaml), ["physics_assessment_in_window"])

    assert rows["doc-pass"]["physics_assessment_in_window"] == "PASS"
    assert rows["doc-none"]["physics_assessment_in_window"] == "NOT_APPLICABLE"


def test_subject_columns_and_sibling_measures_ride_along_aggregated(con):
    rows = _by_document(
        _rules(con, _PRECOMPUTED_RULES_YAML), ["physicist_initials", "assessments"]
    )

    assert rows["doc-pass"] == {"physicist_initials": "AB", "assessments": 2}
    assert rows["doc-fail"] == {"physicist_initials": "CD", "assessments": 1}
    assert rows["doc-none"]["assessments"] == 0


def test_a_boolean_aggregate_status_maps_to_the_tri_state(con):
    rows = _by_document(_rules(con, _BOOLEAN_RULES_YAML), ["any_in_window"])

    assert rows["doc-pass"]["any_in_window"] == "PASS"
    assert rows["doc-fail"]["any_in_window"] == "FAIL"
    assert rows["doc-none"]["any_in_window"] == "PENDING"


def test_rule_metadata_survives_aggregation(con):
    model = _rules(con, _PRECOMPUTED_RULES_YAML)

    extensions = model.get_dimensions()["physics_assessment_in_window"].metadata["ossie"][
        "field_extensions"
    ]["RULES_ENGINE"]
    assert extensions["rule_id"] == "MS-015"
    assert extensions["rule_evaluation_grain"] == "subject"
    assert extensions["rule_subject_columns"] == ["physicist_initials", "assessments"]


def test_the_anchor_stays_an_entity_dimension(con):
    model = _rules(con, _PRECOMPUTED_RULES_YAML)

    assert model.get_dimensions()["sk_plan_document"].is_entity


def test_row_and_subject_grain_statuses_cannot_mix(con):
    with pytest.raises(ConversionError, match="mix"):
        _rules(con, _MIXED_GRAIN_RULES_YAML)


def test_rules_chain_onto_a_model_with_a_computed_dimension(con):
    """A source model may declare dimensions as expressions rather than columns;
    the chained projection materialises them so rule fields can read them."""
    from boring_semantic_layer.ops import Dimension

    evidence = _evidence(con).with_dimensions(
        has_initials=Dimension(expr=lambda table: table.physicist_initials.notnull())
    )
    rules_yaml = _RULES_HEADER + """\
  - name: initials_present
    expr: source.has_initials
    tags:
      - semantic_role:rule_status
      - rule_id:MS-017
      - rule_severity:LOW
      - rule_family_id:ms_physics
      - rule_subject_columns:physicist_initials
      - rule_plain_english:Initials must be present.
"""
    model = evaluate_rules(
        convert_ossie_to_bsl(
            convert_rules_to_ossie(rules_yaml, model_name="ms_physics"), source_model=evidence
        )
    )

    rows = _by_document(model, ["initials_present"])
    assert rows["doc-pass"]["initials_present"] == "PASS"
    assert rows["doc-none"]["initials_present"] == "FAIL"
