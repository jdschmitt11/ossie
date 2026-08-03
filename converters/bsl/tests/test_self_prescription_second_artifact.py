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

"""Second artifact through ossie_bsl: proves the converter generalizes."""

import ibis
import pytest

from ossie_bsl import convert_ossie_to_bsl

from _util import prescription_evidence_ossie, prescription_rules_ossie

# Mirrors the real gold_prescription_with_sk schema (patient_context_graph.duckdb).
SCHEMA = """
    sk_prescription        BIGINT,
    _data_source            VARCHAR,
    site_name                VARCHAR,
    technique                 VARCHAR,
    modality                   VARCHAR,
    fractions                   BIGINT,
    fraction_dose                 DOUBLE,
    total_dose                     DOUBLE,
    target_coverage                 VARCHAR,
    coverage_dose_spec                DOUBLE,
    target_coverage_units               VARCHAR,
    status                                BIGINT,
    approval_dt_tm                        TIMESTAMP,
    approval_staff_id                      BIGINT,
    staff_role                              BIGINT,
    physician_last_name                      VARCHAR,
    physician_initials                        VARCHAR,
    dose_ttl_cum                               DOUBLE,
    start_dt_tm                                 VARCHAR,
    last_dt_tm                                   VARCHAR,
    notes                                         VARCHAR
"""

ROWS = """
    (1, 'MOSAIQ', 'BRAIN', 'IMRT', 'PHOTON', 30, 200.0, 6000.0, '95%', 5.0, 'cm',
     5, TIMESTAMP '2026-01-01 08:00:00', 42, 1, 'Smith', 'JS', 6000.0,
     '2026-01-02', '2026-02-15', 'ok'),
    (2, 'MOSAIQ', '', '', '', 0, 0.0, 0.0, '', NULL, '',
     0, NULL, NULL, NULL, NULL, NULL, 0.0, '', '', ''),
    (3, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
     NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)
"""


@pytest.fixture
def con():
    connection = ibis.duckdb.connect()
    connection.raw_sql(f"CREATE TABLE gold_prescription_with_sk ({SCHEMA})")
    connection.raw_sql(f"INSERT INTO gold_prescription_with_sk VALUES {ROWS}")
    return connection


@pytest.fixture
def evidence(con):
    return convert_ossie_to_bsl(
        prescription_evidence_ossie(),
        tables={"gold_prescription_with_sk": con.table("gold_prescription_with_sk")},
    )


def test_evidence_converts_and_evaluates(evidence):
    rows = (
        evidence.query(dimensions=["sk_prescription", "status_label", "staff_role_label"])
        .execute()
        .set_index("sk_prescription")
        .to_dict("index")
    )
    assert rows[1]["status_label"] == "APPROVE"
    assert rows[1]["staff_role_label"] == "Physician"


def test_rules_chain_and_a_rule_id_that_differs_from_its_dimension_name(evidence):
    predicates = convert_ossie_to_bsl(prescription_rules_ossie(), source_model=evidence)

    rows = (
        predicates.query(dimensions=["prescription_id", "rx_depth_present"])
        .execute()
        .set_index("prescription_id")
    )
    assert bool(rows.loc[1, "rx_depth_present"]) is True
    assert bool(rows.loc[2, "rx_depth_present"]) is False

    extensions = predicates.get_dimensions()["rx_depth_present"].metadata["ossie"][
        "field_extensions"
    ]
    assert extensions["RULES_ENGINE"]["rule_id"] == "MS-007"
