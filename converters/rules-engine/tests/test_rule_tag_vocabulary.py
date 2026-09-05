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

"""Rule tags authored in production that the converter must carry, not reject.

Every tag here appears in a real radonc rule family. An unsupported tag raises
ConversionError by design, so each of these was a hard block on converting that
family through Ossie.
"""

import json

import yaml
from ossie import OSIDocument

from ossie_rules_engine import convert_rules_to_ossie

_HEADER = """\
version: '1.1'
rule_family_version: '1.0.0'
evidence_id: 'self.plan_evidence'
evidence_version: '1.0.0'
render_hint: presence
source: radonc.evidence.self_plan_evidence
use_cases: [patient]
dimensions:
  - name: plan_id
    expr: source.sk_plan
    tags: [rules_engine:anchor, node_type:plan, semantic_role:entity_key]
"""

_RULE_TAGS = """\
      - semantic_role:rule_status
      - rule_id:DC-021
      - rule_severity:HIGH
      - rule_family_id:dc_laterality
      - rule_subject_columns:plan_laterality,dx_laterality
      - rule_plain_english:Laterality must agree.
"""


def _fields(rules_yaml: str, model_name: str = "dc_laterality"):
    document = OSIDocument.model_validate(
        yaml.safe_load(convert_rules_to_ossie(rules_yaml, model_name=model_name))
    )
    return {f.name: f for f in document.semantic_model[0].datasets[0].fields}


def _rules_engine_extension(field):
    return {
        vendor: json.loads(ext.data)
        for vendor, ext in ((e.vendor_name, e) for e in field.custom_extensions)
    }["RULES_ENGINE"]


def test_precomputed_rule_status_encoding_is_carried_into_the_extension():
    """dc_laterality authors an already-encoded status column."""
    rules_yaml = _HEADER + f"""\
  - name: laterality_agrees
    expr: source.laterality_status
    tags:
{_RULE_TAGS}      - rule_status_encoding:precomputed
"""

    payload = _rules_engine_extension(_fields(rules_yaml)["laterality_agrees"])

    assert payload["rule_status_encoding"] == "precomputed"


def test_observation_roles_are_carried_as_a_structured_mapping():
    """self_plan_ml_review binds ML observation columns to named roles."""
    rules_yaml = _HEADER + f"""\
  - name: regimen_rarity_within_threshold
    expr: source.ml_regimen_rarity_score < 0.9
    tags:
{_RULE_TAGS}      - rule_observation_roles:score=ml_score,scoring_status=ml_status
      - rule_threshold_ref:plan_rarity_thresholds@1.0.0
"""

    payload = _rules_engine_extension(
        _fields(rules_yaml, "self_plan_ml_review")["regimen_rarity_within_threshold"]
    )

    assert payload["rule_observation_roles"] == {
        "score": "ml_score",
        "scoring_status": "ml_status",
    }
    assert payload["rule_threshold_ref"] == "plan_rarity_thresholds@1.0.0"


def test_subject_evaluation_grain_is_carried_into_the_extension():
    """ms_physics_assessment_window evaluates one status per subject."""
    rules_yaml = _HEADER + f"""\
  - name: assessment_in_window
    expr: source.in_window
    tags:
{_RULE_TAGS}      - rule_evaluation_grain:subject
"""

    payload = _rules_engine_extension(_fields(rules_yaml)["assessment_in_window"])

    assert payload["rule_evaluation_grain"] == "subject"


def _metrics(rules_yaml: str, model_name: str):
    document = OSIDocument.model_validate(
        yaml.safe_load(convert_rules_to_ossie(rules_yaml, model_name=model_name))
    )
    return {m.name: m for m in (document.semantic_model[0].metrics or [])}


_SUBJECT_GRAIN_FAMILY = _HEADER + """\
measures:
  - name: physicist_sign_date_time
    expr: MAX(source.physicist_sign_date_time)
    display_name: Physicist Sign Date Time
  - name: physics_assessment_in_window
    expr: "CASE WHEN COUNT(source.physicist_sign_date_time) = 0 THEN 'NOT_APPLICABLE' ELSE 'PASS' END"
    display_name: Physics assessment completed within window
    tags:
      - semantic_role:rule_status
      - rule_status_encoding:precomputed
      - rule_evaluation_grain:subject
      - rule_id:MS-015
      - rule_severity:HIGH
      - rule_family_id:ms_physics_assessment_window
      - rule_subject_columns:physicist_sign_date_time
      - rule_plain_english:A physics assessment must complete within the window.
"""


def test_a_rule_status_authored_as_a_measure_satisfies_the_rule_requirement():
    """ms_physics_assessment_window evaluates at subject grain, so its status is
    an aggregate measure rather than a row-level dimension. It is still a rule."""
    metrics = _metrics(_SUBJECT_GRAIN_FAMILY, "ms_physics_assessment_window")

    assert set(metrics) == {"physicist_sign_date_time", "physics_assessment_in_window"}


def test_a_rule_status_measure_carries_its_rule_metadata():
    metric = _metrics(_SUBJECT_GRAIN_FAMILY, "ms_physics_assessment_window")[
        "physics_assessment_in_window"
    ]

    payload = _rules_engine_extension(metric)

    assert payload["semantic_role"] == "rule_status"
    assert payload["rule_id"] == "MS-015"
    assert payload["rule_evaluation_grain"] == "subject"
    assert payload["rule_subject_columns"] == ["physicist_sign_date_time"]


def test_a_family_whose_only_rule_is_a_measure_is_not_rejected():
    convert_rules_to_ossie(_SUBJECT_GRAIN_FAMILY, model_name="ms_physics_assessment_window")


def test_a_family_with_no_rule_status_anywhere_still_raises():
    import pytest

    no_rules = _HEADER + """\
measures:
  - name: physicist_sign_date_time
    expr: MAX(source.physicist_sign_date_time)
"""

    with pytest.raises(Exception, match="at least one rule_status"):
        convert_rules_to_ossie(no_rules, model_name="no_rules")
