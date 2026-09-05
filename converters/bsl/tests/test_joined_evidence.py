"""UC joined evidence executes through OSSIE and BSL."""

from __future__ import annotations

import json
from pathlib import Path

import ibis
import pandas as pd
import yaml
from ossie import OSIDocument

from ossie_bsl import convert_ossie_to_bsl
from ossie_bsl.converter import _joined_expression
from ossie_databricks import convert_metric_view_to_ossie
from ossie_rules_engine import convert_evidence_to_ossie


ARTIFACT = (
    Path(__file__).parents[4]
    / "radonc_semantics/src/radonc_semantics/analytics/artifacts/evidence"
    / "plan_site_setup_context.metric.yaml"
)

FIELDS = [
    "sk_plan",
    "sk_site_setup",
    "setup_data_source",
    "plan_treatment_orientation",
    "setup_patient_orientation",
    "setup_status",
    "setup_sign_date_time",
]


def test_joined_artifact_matches_its_sql_and_applies_the_filter():
    con = ibis.duckdb.connect()
    con.raw_sql(
        "CREATE TABLE gold_edge_plan_to_site_setup AS "
        "SELECT * FROM (VALUES "
        "('varian-plan', 10, 'varian_dwh'), "
        "('mosaiq-plan', 20, 'mosaiq_live'), "
        "('cyberknife-plan', 30, 'cyberknife')"
        ") AS t(sk_plan, sk_site_setup, _data_source)"
    )
    con.raw_sql(
        "CREATE TABLE gold_approved_non_qa_plans AS "
        "SELECT * FROM (VALUES "
        "('varian-plan', 'varian_dwh', 'HFS'), "
        "('mosaiq-plan', 'mosaiq_live', 'HFP'), "
        "('cyberknife-plan', 'cyberknife', NULL)"
        ") AS t(sk_plan, _data_source, treatment_orientation)"
    )
    con.raw_sql(
        "CREATE TABLE gold_site_setup_with_sk AS "
        "SELECT * FROM (VALUES "
        "(10, 'mosaiq_live', 'HFS', '5', '2026-01-01'), "
        "(20, 'mosaiq_live', 'HFP', '6', '2026-01-02'), "
        "(30, 'mosaiq_live', 'FFS', '7', '2026-01-03')"
        ") AS t(sk_site_setup, _data_source, patient_orientation, status, sign_date_time)"
    )

    ossie_yaml = convert_evidence_to_ossie(
        ARTIFACT.read_text(),
        model_name="plan_site_setup_context",
        evidence_id="plan.site_setup_context",
    )
    model = convert_ossie_to_bsl(
        ossie_yaml,
        tables={
            "gold_edge_plan_to_site_setup": con.table("gold_edge_plan_to_site_setup"),
            "gold_approved_non_qa_plans": con.table("gold_approved_non_qa_plans"),
            "gold_site_setup_with_sk": con.table("gold_site_setup_with_sk"),
        },
    )

    assert set(model.table.columns) == set(FIELDS)

    actual = model.query(dimensions=FIELDS).execute().sort_values(FIELDS).reset_index(drop=True)
    expected = con.sql(
        "SELECT e.sk_plan, e.sk_site_setup, "
        "s._data_source AS setup_data_source, p.treatment_orientation "
        "AS plan_treatment_orientation, s.patient_orientation "
        "AS setup_patient_orientation, TRY_CAST(s.status AS BIGINT) AS setup_status, "
        "s.sign_date_time AS setup_sign_date_time "
        "FROM gold_edge_plan_to_site_setup e "
        "LEFT JOIN gold_approved_non_qa_plans p ON e.sk_plan = p.sk_plan "
        "LEFT JOIN gold_site_setup_with_sk s ON e.sk_site_setup = s.sk_site_setup "
        "WHERE e._data_source NOT IN ('mosaiq_live', 'cyberknife', 'ck')"
    ).execute().sort_values(FIELDS).reset_index(drop=True)

    pd.testing.assert_frame_equal(actual, expected)


def test_nested_join_paths_resolve_in_fields_and_model_filter():
    con = ibis.duckdb.connect()
    con.raw_sql("CREATE TABLE root AS SELECT * FROM (VALUES (1, 10), (2, 20)) t(id, edge_id)")
    con.raw_sql("CREATE TABLE edge AS SELECT * FROM (VALUES (10, 100), (20, 200)) t(id, child_id)")
    con.raw_sql(
        "CREATE TABLE child AS SELECT * FROM (VALUES (100, 'alpha', TRUE), (200, 'beta', FALSE)) "
        "t(id, label, is_active)"
    )
    ossie_yaml = convert_metric_view_to_ossie(
        """
version: '1.1'
source: SELECT * FROM root
filter: edge.child.is_active
joins:
  - name: edge
    source: SELECT * FROM edge
    'on': source.edge_id = edge.id
    joins:
      - name: child
        source: SELECT * FROM child
        'on': edge.child_id = child.id
fields:
  - name: id
    expr: source.id
  - name: label
    expr: UPPER(edge.child.label)
""",
        model_name="nested",
    )

    model = convert_ossie_to_bsl(
        ossie_yaml,
        tables={"SELECT * FROM root": con.table("root"), "SELECT * FROM edge": con.table("edge"), "SELECT * FROM child": con.table("child")},
    )

    assert model.query(dimensions=["id", "label"]).execute().sort_values("id").to_dict("records") == [
        {"id": 1, "label": "ALPHA"}
    ]


def test_deep_nested_join_paths_resolve_without_rewriting_string_literals():
    con = ibis.duckdb.connect()
    con.raw_sql("CREATE TABLE root AS SELECT 1 AS id, 10 AS first_id")
    con.raw_sql("CREATE TABLE first AS SELECT 10 AS id, 20 AS second_id")
    con.raw_sql("CREATE TABLE second AS SELECT 20 AS id, 30 AS third_id")
    con.raw_sql("CREATE TABLE third AS SELECT 30 AS id, 40 AS leaf_id")
    con.raw_sql("CREATE TABLE leaf AS SELECT 40 AS id, 'ready' AS status")
    ossie_yaml = convert_metric_view_to_ossie(
        """
version: '1.1'
source: SELECT * FROM root
joins:
  - name: first
    source: SELECT * FROM first
    'on': source.first_id = first.id
    joins:
      - name: second
        source: SELECT * FROM second
        'on': first.second_id = second.id
        joins:
          - name: third
            source: SELECT * FROM third
            'on': second.third_id = third.id
            joins:
              - name: leaf
                source: SELECT * FROM leaf
                'on': third.leaf_id = leaf.id
fields:
  - name: id
    expr: source.id
  - name: status
    expr: first.second.third.leaf.status
""",
        model_name="deep_nested",
    )

    model = convert_ossie_to_bsl(
        ossie_yaml,
        tables={
            "SELECT * FROM root": con.table("root"),
            "SELECT * FROM first": con.table("first"),
            "SELECT * FROM second": con.table("second"),
            "SELECT * FROM third": con.table("third"),
            "SELECT * FROM leaf": con.table("leaf"),
        },
    )

    assert model.query(dimensions=["id", "status"]).execute().to_dict("records") == [
        {"id": 1, "status": "ready"}
    ]


def test_deep_join_path_rewriter_binds_columns_without_touching_literals():
    rewritten = _joined_expression(
        "CASE WHEN first.second.third.leaf.status = "
        "'first.second.third.leaf.status' THEN first.second.third.leaf.status END",
        dataset_name="source",
        alias_indexes={"source": 0, "first.second.third.leaf": 4},
    )

    assert "__ossie_raw__4__status" in rewritten
    assert "'first.second.third.leaf.status'" in rewritten


def test_joined_expression_rewrites_columns_without_corrupting_string_literals():
    con = ibis.duckdb.connect()
    con.raw_sql("CREATE TABLE root AS SELECT 1 AS id, 10 AS child_id")
    con.raw_sql(
        "CREATE TABLE child AS "
        "SELECT 10 AS id, 'child.status is odd' AS status, 'kept' AS label"
    )
    ossie_yaml = convert_metric_view_to_ossie(
        """
version: '1.1'
source: SELECT * FROM root
joins:
  - name: child
    source: SELECT * FROM child
    'on': source.child_id = child.id
fields:
  - name: id
    expr: source.id
  - name: literal_sensitive_label
    expr: CASE WHEN child.status = 'child.status is odd' THEN child.label ELSE NULL END
""",
        model_name="literal_sensitive",
    )

    model = convert_ossie_to_bsl(
        ossie_yaml,
        tables={
            "SELECT * FROM root": con.table("root"),
            "SELECT * FROM child": con.table("child"),
        },
    )

    assert model.query(
        dimensions=["id", "literal_sensitive_label"]
    ).execute().to_dict("records") == [
        {"id": 1, "literal_sensitive_label": "kept"}
    ]


def test_joined_relationship_filter_is_applied_before_the_left_join():
    con = ibis.duckdb.connect()
    con.raw_sql("CREATE TABLE root AS SELECT * FROM (VALUES (1, 10), (2, 20)) t(id, child_id)")
    con.raw_sql(
        "CREATE TABLE child AS "
        "SELECT * FROM (VALUES (10, 'kept', TRUE), (20, 'removed', FALSE)) t(id, label, is_active)"
    )
    document = yaml.safe_load(
        convert_metric_view_to_ossie(
            """
version: '1.1'
source: SELECT * FROM root
joins:
  - name: child
    source: SELECT * FROM child
    'on': source.child_id = child.id
fields:
  - name: id
    expr: source.id
  - name: label
    expr: child.label
""",
            model_name="filtered_join",
        )
    )
    relationship = document["semantic_model"][0]["relationships"][0]
    relationship["custom_extensions"] = [
        {
            "vendor_name": "RULES_ENGINE",
            "data": json.dumps({"_v": 1, "join_filter": "child.is_active"}),
        }
    ]

    model = convert_ossie_to_bsl(
        yaml.safe_dump(document),
        tables={
            "SELECT * FROM root": con.table("root"),
            "SELECT * FROM child": con.table("child"),
        },
    )

    actual = model.query(dimensions=["id", "label"]).execute().sort_values("id")
    assert actual.iloc[0].to_dict() == {"id": 1, "label": "kept"}
    assert actual.iloc[1]["id"] == 2
    assert pd.isna(actual.iloc[1]["label"])


def test_simple_sum_measure_executes_through_ossie_and_bsl():
    con = ibis.duckdb.connect()
    con.raw_sql(
        "CREATE TABLE facts AS "
        "SELECT * FROM (VALUES ('a', 2), ('a', 3), ('b', 5)) t(category, amount)"
    )
    ossie_yaml = convert_metric_view_to_ossie(
        """
version: '1.1'
source: SELECT * FROM facts
dimensions:
  - name: category
    expr: source.category
measures:
  - name: total_amount
    expr: SUM(source.amount)
""",
        model_name="facts",
    )

    model = convert_ossie_to_bsl(
        ossie_yaml,
        tables={"SELECT * FROM facts": con.table("facts")},
    )

    assert "amount" not in model.table.columns

    actual = model.query(
        dimensions=["category"], measures=["total_amount"]
    ).execute().sort_values("category").reset_index(drop=True)

    assert actual.to_dict("records") == [
        {"category": "a", "total_amount": 5},
        {"category": "b", "total_amount": 5},
    ]


def test_distinct_count_measure_executes_through_ossie_and_bsl():
    con = ibis.duckdb.connect()
    con.raw_sql(
        "CREATE TABLE facts AS "
        "SELECT * FROM (VALUES ('a', 1), ('a', 1), ('a', 2), ('b', 2)) t(category, amount)"
    )
    ossie_yaml = convert_metric_view_to_ossie(
        """
version: '1.1'
source: SELECT * FROM facts
dimensions:
  - name: category
    expr: source.category
measures:
  - name: distinct_amounts
    expr: COUNT(DISTINCT source.amount)
""",
        model_name="facts",
    )

    model = convert_ossie_to_bsl(
        ossie_yaml,
        tables={"SELECT * FROM facts": con.table("facts")},
    )

    actual = model.query(
        dimensions=["category"], measures=["distinct_amounts"]
    ).execute().sort_values("category").reset_index(drop=True)

    assert actual.to_dict("records") == [
        {"category": "a", "distinct_amounts": 2},
        {"category": "b", "distinct_amounts": 1},
    ]


def test_conditional_distinct_count_measure_executes_through_ossie_and_bsl():
    con = ibis.duckdb.connect()
    con.raw_sql(
        "CREATE TABLE facts AS SELECT * FROM (VALUES "
        "('a', ' treatment ', 1), ('a', 'TREATMENT', 1), ('a', 'image_guidance', 2), "
        "('b', NULL, 3), ('b', 'treatment', 4)"
        ") t(category, kind, charge)"
    )
    ossie_yaml = convert_metric_view_to_ossie(
        """
version: '1.1'
source: SELECT * FROM facts
dimensions:
  - name: category
    expr: source.category
measures:
  - name: treatment_charges
    expr: |-
      COUNT(DISTINCT CASE
        WHEN LOWER(TRIM(COALESCE(source.kind, ''))) = 'treatment'
        THEN source.charge
      END)
""",
        model_name="facts",
    )

    model = convert_ossie_to_bsl(
        ossie_yaml,
        tables={"SELECT * FROM facts": con.table("facts")},
    )

    actual = model.query(
        dimensions=["category"], measures=["treatment_charges"]
    ).execute().sort_values("category").reset_index(drop=True)

    assert actual.to_dict("records") == [
        {"category": "a", "treatment_charges": 1},
        {"category": "b", "treatment_charges": 1},
    ]


def test_joined_conditional_distinct_count_uses_declared_dataset_path():
    con = ibis.duckdb.connect()
    con.raw_sql("CREATE TABLE appointment AS SELECT * FROM (VALUES (1)) t(id)")
    con.raw_sql(
        "CREATE TABLE appointment_volume AS "
        "SELECT * FROM (VALUES (1, 10), (1, 11)) t(appointment_id, volume_id)"
    )
    con.raw_sql(
        "CREATE TABLE volume AS SELECT * FROM (VALUES "
        "(10, ' treatment ', 'technical', 100), "
        "(11, 'treatment', 'professional', 101)"
        ") t(id, activity_subgroup, procedure_code_type, charge_activity_id)"
    )
    ossie_yaml = convert_metric_view_to_ossie(
        """
version: '1.1'
source: SELECT * FROM appointment
joins:
  - name: appointment_volume
    source: SELECT * FROM appointment_volume
    'on': source.id = appointment_volume.appointment_id
    joins:
      - name: volume
        source: SELECT * FROM volume
        'on': appointment_volume.volume_id = volume.id
dimensions:
  - name: appointment_id
    expr: source.id
measures:
  - name: daily_tx_charge_count
    expr: >-
      COUNT(DISTINCT CASE
        WHEN lower(trim(coalesce(appointment_volume.volume.activity_subgroup, ''))) = 'treatment'
             AND lower(trim(coalesce(appointment_volume.volume.procedure_code_type, ''))) IN ('technical', 'pending')
        THEN appointment_volume.volume.charge_activity_id
      END)
""",
        model_name="appointment_monitoring_summary",
    )

    model = convert_ossie_to_bsl(
        ossie_yaml,
        tables={
            "SELECT * FROM appointment": con.table("appointment"),
            "SELECT * FROM appointment_volume": con.table("appointment_volume"),
            "SELECT * FROM volume": con.table("volume"),
        },
    )

    assert model.query(
        dimensions=["appointment_id"], measures=["daily_tx_charge_count"]
    ).execute().to_dict("records") == [
        {"appointment_id": 1, "daily_tx_charge_count": 1}
    ]


def test_laterality_aggregate_measures_execute_through_ossie_and_bsl():
    con = ibis.duckdb.connect()
    tables = {
        "gold_approved_non_qa_plans": {"sk_plan": ["p1"]},
        "gold_edge_prescription_to_plan": {"sk_plan": ["p1"], "sk_prescription": ["rx1"]},
        "gold_prescription_with_sk": {"sk_prescription": ["rx1"]},
        "gold_edge_course_to_prescription": {"sk_prescription": ["rx1"], "sk_treatment_course": ["c1"]},
        "gold_edge_diagnosis_to_course": {"sk_treatment_course": ["c1"], "sk_diagnosis": ["dx1"]},
        "gold_diagnosis_with_sk": {"sk_diagnosis": ["dx1"], "_data_source": ["mosaiq_live"], "is_primary": ["Y"]},
        "gold_edge_plan_to_varian_field": {"sk_plan": ["p1", "p1"], "sk_varian_treatment_field": ["f1", "f2"]},
        "gold_edge_varian_field_to_varian_details": {"sk_varian_treatment_field": ["f1", "f2"], "sk_field_details": ["d1", "d2"]},
        "gold_varian_treatment_field_details_with_sk": {"sk_field_details": ["d1", "d2"], "iso_x_mm": [-30.0, 25.0]},
    }
    for name, data in tables.items():
        con.create_table(name, ibis.memtable(data))

    artifact = (
        Path(__file__).parents[4]
        / "radonc_semantics/src/radonc_semantics/analytics/artifacts/evidence"
        / "laterality_context.metric.yaml"
    )
    evidence = yaml.safe_load(artifact.read_text())
    evidence["fields"] = [field for field in evidence["fields"] if field["name"] == "sk_plan"]
    ossie_yaml = convert_evidence_to_ossie(
        yaml.safe_dump(evidence, sort_keys=False),
        model_name="laterality_context",
        evidence_id="laterality.context",
    )
    document = OSIDocument.model_validate(yaml.safe_load(ossie_yaml))
    bound_tables = {
        dataset.source: (
            con.sql(dataset.source)
            if dataset.source.lstrip().upper().startswith(("SELECT", "WITH"))
            else con.table(dataset.source)
        )
        for dataset in document.semantic_model[0].datasets
    }

    model = convert_ossie_to_bsl(ossie_yaml, tables=bound_tables)

    assert set(model.get_measures()) == {"min_iso_x_mm", "max_iso_x_mm"}
    assert set(model.get_calculated_measures()) == {
        "plan_eclipse_laterality",
        "iso_laterality_applicable",
    }

    assert model.query(
        dimensions=["sk_plan"],
        measures=[
            "min_iso_x_mm",
            "max_iso_x_mm",
            "plan_eclipse_laterality",
            "iso_laterality_applicable",
        ],
    ).execute().to_dict("records") == [
        {
            "sk_plan": "p1",
            "min_iso_x_mm": -30.0,
            "max_iso_x_mm": 25.0,
            "plan_eclipse_laterality": "Ambiguous",
            "iso_laterality_applicable": True,
        }
    ]
