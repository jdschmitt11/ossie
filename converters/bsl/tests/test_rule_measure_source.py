"""Explicit metric-view rule sources compile through BSL's query API."""

from __future__ import annotations

import ibis
import pytest
from boring_semantic_layer import to_semantic_table
from boring_semantic_layer.ops import Dimension

from ossie_bsl.internals import ConversionError, query_rule_source_model


@pytest.fixture
def evidence():
    con = ibis.duckdb.connect()
    con.create_table(
        "appointments",
        ibis.memtable(
            {
                "appointment_id": ["a1", "a1", "a2"],
                "appointment_status": ["C", "C", "X"],
                "charge": [1, 1, 1],
            }
        ),
    )
    return to_semantic_table(con.table("appointments")).with_dimensions(
        appointment_id=Dimension(expr=lambda table: table.appointment_id, is_entity=True),
        appointment_status=lambda table: table.appointment_status,
    ).with_measures(daily_tx_charge_count=lambda table: table.charge.sum())


def test_metric_view_rule_source_groups_declared_dimensions_and_measures(evidence):
    model = query_rule_source_model(
        evidence,
        source="""
            SELECT appointment_id, appointment_status,
                   MEASURE(daily_tx_charge_count) AS daily_tx_charge_count
            FROM radonc.evidence.appointment_monitoring_summary
            GROUP BY appointment_id, appointment_status
        """,
        primary_key=["appointment_id"],
    )

    rows = model.query(
        dimensions=["appointment_id", "appointment_status", "daily_tx_charge_count"]
    ).execute().sort_values("appointment_id").reset_index(drop=True)

    assert rows.to_dict("records") == [
        {"appointment_id": "a1", "appointment_status": "C", "daily_tx_charge_count": 2},
        {"appointment_id": "a2", "appointment_status": "X", "daily_tx_charge_count": 1},
    ]


def test_metric_view_rule_source_rejects_undeclared_filtering(evidence):
    with pytest.raises(ConversionError, match="no WITH, JOIN, or WHERE"):
        query_rule_source_model(
            evidence,
            source="""
                SELECT appointment_id, MEASURE(daily_tx_charge_count) AS daily_tx_charge_count
                FROM radonc.evidence.appointment_monitoring_summary
                WHERE appointment_status = 'C'
                GROUP BY appointment_id
            """,
            primary_key=["appointment_id"],
        )
