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
from pathlib import Path

import yaml
from ossie import OSIDialect, OSIDocument

from ossie_rules_engine import convert_evidence_to_ossie

from _util import load_fixture

EXPECTED_FIELDS = [
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
    "total_dose_cgy",
    "dose_per_fraction_cgy",
    "coverage_percent",
    "treatment_orientation",
    "tx_site",
    "tx_site_classified",
    "laterality",
    "treatment_type",
    "is_replan",
]


def _convert():
    return convert_evidence_to_ossie(
        load_fixture("self_plan_evidence.metric.yaml"),
        model_name="self_plan_evidence",
        evidence_id="self.plan_evidence",
    )


def test_self_plan_evidence_produces_a_schema_valid_ossie_document():
    doc = OSIDocument.model_validate(yaml.safe_load(_convert()))

    assert len(doc.semantic_model) == 1
    model = doc.semantic_model[0]
    assert model.name == "self_plan_evidence"
    assert len(model.datasets) == 1


def test_self_plan_evidence_carries_source_and_every_field():
    dataset = OSIDocument.model_validate(yaml.safe_load(_convert())).semantic_model[0].datasets[0]

    assert dataset.source == "gold_plan_with_sk"
    assert [field.name for field in dataset.fields] == EXPECTED_FIELDS


def test_the_detected_anchor_becomes_the_dataset_primary_key():
    dataset = OSIDocument.model_validate(yaml.safe_load(_convert())).semantic_model[0].datasets[0]

    assert dataset.primary_key == ["sk_plan"]


def test_authored_sql_lands_on_the_databricks_dialect():
    dataset = OSIDocument.model_validate(yaml.safe_load(_convert())).semantic_model[0].datasets[0]
    by_name = {field.name: field for field in dataset.fields}

    dialects = by_name["number_of_fractions"].expression.dialects
    assert [entry.dialect for entry in dialects] == [OSIDialect.DATABRICKS]
    assert dialects[0].expression == "TRY_CAST(source.number_of_fractions AS BIGINT)"


def test_display_name_and_comment_map_to_label_and_description():
    dataset = OSIDocument.model_validate(yaml.safe_load(_convert())).semantic_model[0].datasets[0]
    by_name = {field.name: field for field in dataset.fields}

    assert by_name["number_of_fractions"].label == "Number of Fractions"
    assert by_name["total_dose_cgy"].description == (
        "Governed model-input alias for the source total dose, which is stored in cGy."
    )
    assert by_name["_data_source"].description is None


def test_model_comment_is_preserved_verbatim_as_the_model_description():
    model = OSIDocument.model_validate(yaml.safe_load(_convert())).semantic_model[0]

    assert model.description == (
        "display_name: Plan Evidence\n"
        "description: Plan-level completeness, presence, regimen, and governed ML "
        "review fields. Anchored at plan."
    )


def test_artifact_identity_lands_in_a_versioned_model_extension():
    model = OSIDocument.model_validate(yaml.safe_load(_convert())).semantic_model[0]

    extensions = {ext.vendor_name: json.loads(ext.data) for ext in model.custom_extensions}
    assert extensions["RULES_ENGINE"] == {
        "_v": 1,
        "artifact_kind": "evidence",
        "source_format": "rules_engine_metric_yaml",
        "source_format_version": "1.1",
        "evidence_id": "self.plan_evidence",
        "evidence_version": "1.2.0",
        "anchor_node_type": "plan",
    }


def test_anchor_field_keeps_its_structured_tag_extension():
    dataset = OSIDocument.model_validate(yaml.safe_load(_convert())).semantic_model[0].datasets[0]
    by_name = {field.name: field for field in dataset.fields}

    extensions = {
        ext.vendor_name: json.loads(ext.data) for ext in by_name["sk_plan"].custom_extensions
    }
    assert extensions["RULES_ENGINE"] == {
        "_v": 1,
        "anchor": True,
        "node_type": "plan",
        "evidence_version": "1.2.0",
    }
    assert by_name["plan_name"].custom_extensions is None


def _authored_evidence_with_measure(**measure_overrides):
    artifact = (
        Path(__file__).parents[4]
        / "radonc_semantics/src/radonc_semantics/artifacts/evidence"
        / "self_volume_evidence.metric.yaml"
    )
    authored = yaml.safe_load(artifact.read_text())
    measure = {
        "name": "procedure_count",
        "expr": "SUM(source.quantity)",
        **measure_overrides,
    }
    authored["measures"] = [measure]
    return authored


def test_evidence_preserves_its_uc_metric_measure():
    document = OSIDocument.model_validate(
        yaml.safe_load(
            convert_evidence_to_ossie(
                yaml.safe_dump(_authored_evidence_with_measure(), sort_keys=False),
                model_name="self_volume_evidence",
                evidence_id="self.volume_evidence",
            )
        )
    )

    assert [(metric.name, metric.expression.dialects[0].expression) for metric in document.semantic_model[0].metrics] == [
        ("procedure_count", "SUM(source.quantity)")
    ]


def test_measure_synonyms_and_format_map_to_their_native_ossie_homes():
    authored = _authored_evidence_with_measure(
        synonyms=["procedure total", "procedure volume"],
        format="0,0",
    )

    document = OSIDocument.model_validate(
        yaml.safe_load(
            convert_evidence_to_ossie(
                yaml.safe_dump(authored, sort_keys=False),
                model_name="self_volume_evidence",
                evidence_id="self.volume_evidence",
            )
        )
    )
    metric = document.semantic_model[0].metrics[0]

    assert metric.ai_context.synonyms == (
        "procedure total",
        "procedure volume",
    )
    extensions = {
        extension.vendor_name: json.loads(extension.data)
        for extension in metric.custom_extensions
    }
    assert extensions["DATABRICKS"] == {"_v": 1, "format": "0,0"}
