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

"""Second artifact through the converters: proves generality past self-plan.

Surfaces two real gaps self-plan never exercised: an authored ``rule_id`` that
differs from its dimension name (``MS-007``/``MS-008``/``MS-009``), and two
tags self-plan never used (``rule_version``, ``rule_audience_roles``).
"""

import json

import yaml
from ossie import OSIDocument

from ossie_rules_engine import ConversionError, convert_evidence_to_ossie, convert_rules_to_ossie

from _util import load_fixture


def _evidence_ossie():
    return convert_evidence_to_ossie(
        load_fixture("self_prescription_evidence.metric.yaml"),
        model_name="self_prescription_evidence",
        evidence_id="self.prescription_evidence",
    )


def _rules_ossie():
    return convert_rules_to_ossie(
        load_fixture("self_prescription.rules.metric.yaml"), model_name="self_prescription"
    )


def test_self_prescription_evidence_produces_a_schema_valid_document():
    doc = OSIDocument.model_validate(yaml.safe_load(_evidence_ossie()))

    dataset = doc.semantic_model[0].datasets[0]
    assert dataset.source == "gold_prescription_with_sk"
    assert dataset.primary_key == ["sk_prescription"]
    assert len(dataset.fields) == 23


def test_self_prescription_rules_produce_a_schema_valid_document():
    doc = OSIDocument.model_validate(yaml.safe_load(_rules_ossie()))

    dataset = doc.semantic_model[0].datasets[0]
    assert dataset.primary_key == ["prescription_id"]
    assert len(dataset.fields) == 12  # 1 anchor + 11 rules


def test_a_rule_id_that_differs_from_its_dimension_name_is_preserved():
    """MS-007 is the authored rule_id for the dimension named rx_depth_present."""
    dataset = OSIDocument.model_validate(yaml.safe_load(_rules_ossie())).semantic_model[0].datasets[0]
    by_name = {field.name: field for field in dataset.fields}

    extensions = {
        ext.vendor_name: json.loads(ext.data)
        for ext in by_name["rx_depth_present"].custom_extensions
    }
    assert extensions["RULES_ENGINE"]["rule_id"] == "MS-007"


def test_rule_version_is_preserved():
    dataset = OSIDocument.model_validate(yaml.safe_load(_rules_ossie())).semantic_model[0].datasets[0]
    by_name = {field.name: field for field in dataset.fields}

    extensions = {
        ext.vendor_name: json.loads(ext.data)
        for ext in by_name["total_dose_present"].custom_extensions
    }
    assert extensions["RULES_ENGINE"]["rule_version"] == "1.0.0"


def test_rule_audience_roles_is_split_into_a_list():
    dataset = OSIDocument.model_validate(yaml.safe_load(_rules_ossie())).semantic_model[0].datasets[0]
    by_name = {field.name: field for field in dataset.fields}

    extensions = {
        ext.vendor_name: json.loads(ext.data)
        for ext in by_name["rx_physician_signed"].custom_extensions
    }
    assert extensions["RULES_ENGINE"]["rule_audience_roles"] == ["doctor", "physicist"]


def test_a_rule_with_no_audience_roles_tag_has_none_in_the_extension():
    dataset = OSIDocument.model_validate(yaml.safe_load(_rules_ossie())).semantic_model[0].datasets[0]
    by_name = {field.name: field for field in dataset.fields}

    extensions = {
        ext.vendor_name: json.loads(ext.data)
        for ext in by_name["total_dose_present"].custom_extensions
    }
    assert "rule_audience_roles" not in extensions["RULES_ENGINE"]


def test_both_artifacts_still_convert_side_by_side():
    """Regression: self-plan must not regress while self-prescription is added."""
    assert convert_evidence_to_ossie(
        load_fixture("self_plan_evidence.metric.yaml"),
        model_name="self_plan_evidence",
        evidence_id="self.plan_evidence",
    )
    assert convert_rules_to_ossie(load_fixture("self_plan.rules.metric.yaml"), model_name="self_plan")
