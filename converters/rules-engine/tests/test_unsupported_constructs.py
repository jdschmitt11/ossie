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

"""Everything out of pilot scope must raise rather than be silently dropped."""

import json
import pytest
import yaml
from ossie import OSIDocument

from ossie_rules_engine import (
    ConversionError,
    convert_evidence_to_ossie,
    convert_rules_to_ossie,
)

from _util import load_fixture


def _evidence(mutate=None) -> str:
    document = yaml.safe_load(load_fixture("self_plan_evidence.metric.yaml"))
    if mutate is not None:
        mutate(document)
    return convert_evidence_to_ossie(
        yaml.safe_dump(document),
        model_name="self_plan_evidence",
        evidence_id="self.plan_evidence",
    )


def _rules(mutate=None) -> str:
    document = yaml.safe_load(load_fixture("self_plan.rules.metric.yaml"))
    if mutate is not None:
        mutate(document)
    return convert_rules_to_ossie(yaml.safe_dump(document), model_name="self_plan")


def test_the_unmodified_fixtures_convert():
    assert _evidence()
    assert _rules()


def test_top_level_filter_is_preserved_for_non_joined_evidence():
    filter_sql = "source.plan_status = 'ACTIVE'"

    def mutate(document):
        document["filter"] = filter_sql

    converted = _evidence(mutate)
    model = OSIDocument.model_validate(yaml.safe_load(converted)).semantic_model[0]
    metadata = {
        ext.vendor_name: json.loads(ext.data) for ext in model.custom_extensions or []
    }
    assert metadata["DATABRICKS"]["filter"] == filter_sql


@pytest.mark.parametrize("key", ["synonyms", "format", "edge_type", "joins", "source"])
def test_unsupported_field_level_keys_raise(key):
    def mutate(document):
        document["fields"][2][key] = "anything"

    with pytest.raises(ConversionError, match=key):
        _evidence(mutate)


def test_a_malformed_tag_raises():
    def mutate(document):
        document["fields"][0]["tags"].append("no_separator")

    with pytest.raises(ConversionError, match="malformed tag"):
        _evidence(mutate)


def test_an_expression_referencing_an_undeclared_table_raises():
    """A single-source OSSIE dataset cannot hide ambient table dependencies."""

    def mutate(document):
        document["fields"][2]["expr"] = (
            "EXISTS (SELECT 1 FROM gold_other other "
            "WHERE other.sk_plan = source.sk_plan)"
        )

    with pytest.raises(ConversionError, match="undeclared table.*gold_other"):
        _evidence(mutate)


@pytest.mark.parametrize(
    "tag",
    [
        "rule_render_widget:timeline",
        "rule_owner:physics",
    ],
)
def test_rule_tags_outside_the_vocabulary_raise(tag):
    """An unrecognised tag must raise rather than be silently dropped."""

    def mutate(document):
        document["dimensions"][1]["tags"].append(tag)

    with pytest.raises(ConversionError, match="unsupported tag"):
        _rules(mutate)


def test_malformed_observation_roles_raise():
    """rule_observation_roles is a role=column mapping, not a bare list."""

    def mutate(document):
        document["dimensions"][1]["tags"].append("rule_observation_roles:reviewer")

    with pytest.raises(ConversionError, match="expected role=column"):
        _rules(mutate)


def test_a_duplicate_tag_key_raises():
    def mutate(document):
        document["dimensions"][1]["tags"].append("rule_severity:LOW")

    with pytest.raises(ConversionError, match="duplicate tag key"):
        _rules(mutate)


def test_duplicate_field_names_raise():
    def mutate(document):
        document["fields"].append(dict(document["fields"][2]))

    with pytest.raises(ConversionError, match="duplicate field name"):
        _evidence(mutate)


def test_declaring_both_fields_and_dimensions_raises():
    def mutate(document):
        document["dimensions"] = document["fields"]

    with pytest.raises(ConversionError, match="only one of"):
        _evidence(mutate)


def test_declaring_neither_fields_nor_dimensions_raises():
    def mutate(document):
        del document["fields"]

    with pytest.raises(ConversionError, match="'fields' or 'dimensions'"):
        _evidence(mutate)


def test_a_missing_source_raises():
    def mutate(document):
        del document["source"]

    with pytest.raises(ConversionError, match="source"):
        _evidence(mutate)


def test_a_field_without_an_expr_raises():
    def mutate(document):
        del document["fields"][2]["expr"]

    with pytest.raises(ConversionError, match="expr"):
        _evidence(mutate)


@pytest.mark.parametrize("anchor_count", [0, 2])
def test_evidence_needs_exactly_one_anchor(anchor_count):
    def mutate(document):
        if anchor_count == 0:
            document["fields"][0]["tags"].remove("rules_engine:anchor")
        else:
            document["fields"][1]["tags"] = ["rules_engine:anchor", "node_type:plan"]

    with pytest.raises(ConversionError, match="exactly one anchor"):
        _evidence(mutate)


@pytest.mark.parametrize("key", ["rule_severity", "rule_family_id", "rule_plain_english"])
def test_a_rule_missing_required_metadata_raises(key):
    def mutate(document):
        tags = document["dimensions"][1]["tags"]
        document["dimensions"][1]["tags"] = [
            tag for tag in tags if not tag.startswith(f"{key}:")
        ]

    with pytest.raises(ConversionError, match="missing required tags"):
        _rules(mutate)


def test_a_rule_id_that_differs_from_its_dimension_name_is_accepted():
    """rule_id is an independent identity (e.g. "MS-007"); see self-prescription."""

    def mutate(document):
        tags = document["dimensions"][1]["tags"]
        document["dimensions"][1]["tags"] = [
            "rule_id:something_else" if tag.startswith("rule_id:") else tag for tag in tags
        ]

    assert _rules(mutate)


def test_a_rule_family_with_no_rules_raises():
    def mutate(document):
        document["dimensions"] = document["dimensions"][:1]

    with pytest.raises(ConversionError, match="at least one rule_status"):
        _rules(mutate)


def test_a_non_mapping_document_raises():
    with pytest.raises(ConversionError, match="mapping"):
        convert_rules_to_ossie("- not a mapping", model_name="self_plan")
