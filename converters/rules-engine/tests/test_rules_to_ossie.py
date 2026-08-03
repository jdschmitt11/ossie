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

import json

import yaml
from ossie import OSIDialect, OSIDocument

from ossie_rules_engine import convert_rules_to_ossie

from _util import load_fixture

RULE_IDS = [
    "total_dose_present",
    "fractions_present",
    "coverage_percent_present",
    "plan_approved_present",
    "tx_site_present",
    "plan_intent_present",
    "treatment_orientation_present",
]


def _model():
    ossie_yaml = convert_rules_to_ossie(
        load_fixture("self_plan.rules.metric.yaml"), model_name="self_plan"
    )
    return OSIDocument.model_validate(yaml.safe_load(ossie_yaml)).semantic_model[0]


def _extension(obj):
    extensions = {ext.vendor_name: json.loads(ext.data) for ext in obj.custom_extensions}
    return extensions["RULES_ENGINE"]


def test_self_plan_rules_produce_a_schema_valid_ossie_document():
    model = _model()

    assert model.name == "self_plan"
    assert len(model.datasets) == 1
    assert model.datasets[0].source == "radonc.evidence.self_plan_evidence"


def test_authored_dimension_names_are_preserved_exactly():
    dataset = _model().datasets[0]

    assert [field.name for field in dataset.fields] == ["plan_id", *RULE_IDS]


def test_the_anchor_dimension_becomes_the_primary_key():
    assert _model().datasets[0].primary_key == ["plan_id"]


def test_predicate_sql_lands_on_the_databricks_dialect():
    by_name = {field.name: field for field in _model().datasets[0].fields}

    dialects = by_name["coverage_percent_present"].expression.dialects
    assert [entry.dialect for entry in dialects] == [OSIDialect.DATABRICKS]
    assert dialects[0].expression == "TRY_CAST(source.coverage_percent AS DOUBLE) > 0"


def test_family_identity_lands_in_a_versioned_model_extension():
    assert _extension(_model()) == {
        "_v": 1,
        "artifact_kind": "rules",
        "source_format": "rules_engine_metric_yaml",
        "source_format_version": "1.1",
        "rule_family_id": "self_plan",
        "rule_family_version": "1.0.0",
        "evidence_id": "self.plan_evidence",
        "evidence_version": "1.2.0",
        "use_cases": ["patient"],
        "render_hint": "presence",
        "anchor_node_type": "plan",
    }


def test_each_rule_carries_structured_status_metadata():
    by_name = {field.name: field for field in _model().datasets[0].fields}

    assert _extension(by_name["total_dose_present"]) == {
        "_v": 1,
        "semantic_role": "rule_status",
        "rule_id": "total_dose_present",
        "rule_severity": "HIGH",
        "rule_family_id": "self_plan",
        "rule_subject_columns": ["total_dose"],
        "rule_plain_english": "The plan total dose must be greater than zero.",
    }


def test_subject_columns_are_split_into_a_list():
    by_name = {field.name: field for field in _model().datasets[0].fields}

    assert _extension(by_name["plan_approved_present"])["rule_subject_columns"] == [
        "plan_approved",
        "plan_approved_date_time",
        "plan_approved_user",
    ]


def test_plain_english_keeps_its_own_punctuation():
    by_name = {field.name: field for field in _model().datasets[0].fields}

    assert _extension(by_name["plan_intent_present"])["rule_plain_english"] == (
        "The plan must specify an intent (curative, palliative, etc.)."
    )


def test_the_identity_dimension_is_not_a_rule():
    by_name = {field.name: field for field in _model().datasets[0].fields}

    assert _extension(by_name["plan_id"]) == {
        "_v": 1,
        "anchor": True,
        "node_type": "plan",
        "semantic_role": "entity_key",
    }


def test_model_comment_is_preserved_verbatim():
    assert _model().description == (
        "Checks whether core plan fields are present and usable.\n"
        "Display name: Plan Completeness"
    )
