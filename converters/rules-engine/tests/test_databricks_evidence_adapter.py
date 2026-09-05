"""Joined UC Metric YAML is structurally imported by ossie-databricks."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
import pytest
from ossie import OSIDocument

from ossie_rules_engine import ConversionError, convert_evidence_to_ossie
from ossie_rules_engine.evidence import _as_select_source


ARTIFACT = (
    Path(__file__).parents[4]
    / "radonc_semantics/src/radonc_semantics/analytics/artifacts/evidence"
    / "plan_site_setup_context.metric.yaml"
)


def test_joined_evidence_uses_databricks_importer_and_keeps_rules_metadata():
    converted = convert_evidence_to_ossie(
        ARTIFACT.read_text(),
        model_name="plan_site_setup_context",
        evidence_id="plan.site_setup_context",
    )

    model = OSIDocument.model_validate(yaml.safe_load(converted)).semantic_model[0]
    datasets = {dataset.name: dataset for dataset in model.datasets}

    assert {name: dataset.source for name, dataset in datasets.items()} == {
        "plan_site_setup_context": "gold_edge_plan_to_site_setup",
        "plan": "gold_approved_non_qa_plans",
        "site_setup": "gold_site_setup_with_sk",
    }
    assert datasets["plan_site_setup_context"].primary_key == ["sk_plan"]
    assert {
        (
            relationship.from_dataset,
            relationship.to,
            tuple(relationship.from_columns),
            tuple(relationship.to_columns),
        )
        for relationship in model.relationships
    } == {
        ("plan_site_setup_context", "plan", ("sk_plan",), ("sk_plan",)),
        ("plan_site_setup_context", "site_setup", ("sk_site_setup",), ("sk_site_setup",)),
    }

    model_metadata = {
        ext.vendor_name: json.loads(ext.data) for ext in model.custom_extensions
    }
    assert model_metadata["DATABRICKS"]["filter"] == (
        "source._data_source NOT IN ('mosaiq_live', 'cyberknife', 'ck')"
    )

    fields = {field.name: field for field in datasets["plan_site_setup_context"].fields}
    metadata = {ext.vendor_name: json.loads(ext.data) for ext in fields["sk_plan"].custom_extensions}
    assert metadata["RULES_ENGINE"] == {
        "_v": 1,
        "anchor": True,
        "node_type": "plan",
        "evidence_version": "1.0.0",
    }


def test_join_metadata_is_preserved_in_the_ossie_relationship_extension():
    document = yaml.safe_load(ARTIFACT.read_text())
    document["joins"][0]["edge_type"] = "plan_to_site_setup"

    converted = convert_evidence_to_ossie(
        yaml.safe_dump(document),
        model_name="plan_site_setup_context",
        evidence_id="plan.site_setup_context",
    )

    model = OSIDocument.model_validate(yaml.safe_load(converted)).semantic_model[0]
    relationship = next(item for item in model.relationships if item.to == "plan")
    metadata = {
        ext.vendor_name: json.loads(ext.data) for ext in relationship.custom_extensions
    }
    assert metadata["RULES_ENGINE"] == {"_v": 1, "edge_type": "plan_to_site_setup"}


def test_plan_offset_parity_preserves_author_edge_metadata():
    artifact = (
        Path(__file__).parents[4]
        / "radonc_semantics/src/radonc_semantics/analytics/artifacts/evidence"
        / "plan_offset_parity_context.metric.yaml"
    )

    converted = convert_evidence_to_ossie(
        artifact.read_text(),
        model_name="plan_offset_parity_context",
        evidence_id="plan.offset_parity_context",
    )

    model = OSIDocument.model_validate(yaml.safe_load(converted)).semantic_model[0]
    assert {relationship.to for relationship in model.relationships} >= {
        "detail_edge",
        "plan_edge",
    }
    detail_edge = next(item for item in model.relationships if item.to == "detail_edge")
    metadata = {
        ext.vendor_name: json.loads(ext.data) for ext in detail_edge.custom_extensions
    }
    assert metadata["RULES_ENGINE"]["edge_type"] == "plan_to_varian_field"


def test_inline_sql_source_is_not_wrapped_again():
    source = "SELECT * FROM gold_diagnosis_with_sk WHERE is_primary = 'Y'"

    assert _as_select_source(source) == source


def test_laterality_context_preserves_its_child_join_filter_without_rewriting_source():
    artifact = (
        Path(__file__).parents[4]
        / "radonc_semantics/src/radonc_semantics/analytics/artifacts/evidence"
        / "laterality_context.metric.yaml"
    )

    converted = convert_evidence_to_ossie(
        artifact.read_text(),
        model_name="laterality_context",
        evidence_id="laterality.context",
    )

    model = OSIDocument.model_validate(yaml.safe_load(converted)).semantic_model[0]
    dx = next(dataset for dataset in model.datasets if dataset.name == "dx")
    assert dx.source == "gold_diagnosis_with_sk"
    relationship = next(item for item in model.relationships if item.to == "dx")
    metadata = {
        ext.vendor_name: json.loads(ext.data) for ext in relationship.custom_extensions
    }
    assert "dx._data_source" in metadata["RULES_ENGINE"]["join_filter"]


def test_document_physics_assessment_converts_as_a_declared_join():
    """The document/physics-assessment join lives in its own artifact now;
    self_document_evidence is single-source again."""
    artifact = (
        Path(__file__).parents[4]
        / "radonc_semantics/src/radonc_semantics/analytics/artifacts/evidence"
        / "document_physics_assessment_evidence.metric.yaml"
    )

    converted = convert_evidence_to_ossie(
        artifact.read_text(),
        model_name="document_physics_assessment_evidence",
        evidence_id="document.physics_assessment_evidence",
    )
    model = OSIDocument.model_validate(yaml.safe_load(converted)).semantic_model[0]
    assert {relationship.to for relationship in model.relationships} == {'document', 'physics_assessment'}


def test_plan_imrt_charge_summary_converts_as_a_declared_join():
    artifact = (
        Path(__file__).parents[4]
        / "radonc_semantics/src/radonc_semantics/analytics/artifacts/evidence"
        / "plan_imrt_charge_context.metric.yaml"
    )
    converted = convert_evidence_to_ossie(
        artifact.read_text(),
        model_name="plan_imrt_charge_context",
        evidence_id="plan.imrt_charge_context",
    )
    model = OSIDocument.model_validate(yaml.safe_load(converted)).semantic_model[0]
    assert {relationship.to for relationship in model.relationships} == {'ck_field', 'imrt_field', 'mosaiq_field', 'patient_activity', 'varian_field'}
