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

import pytest
import yaml

from ossie_rules_engine import (
    ConversionError,
    convert_rules_to_ossie,
)

from _util import load_fixture


def _rules(mutate=None) -> str:
    document = yaml.safe_load(load_fixture("self_plan.rules.metric.yaml"))
    if mutate is not None:
        mutate(document)
    return convert_rules_to_ossie(yaml.safe_dump(document), model_name="self_plan")


def test_the_unmodified_fixtures_convert():
    assert _rules()


@pytest.mark.parametrize("key,value", [
    ("filter", "source.sk_plan IS NOT NULL"),
    ("joins", [{"name": "other", "source": "gold_other", "on": "source.id = other.id"}]),
])
def test_top_level_filter_and_joins_are_rejected(key, value):
    with pytest.raises(ConversionError, match="is not supported yet"):
        _rules(lambda document: document.update({key: value}))


def test_a_malformed_tag_raises():
    def mutate(document):
        document["dimensions"][1]["tags"].append("no_separator")

    with pytest.raises(ConversionError, match="malformed tag"):
        _rules(mutate)


@pytest.mark.parametrize("key", ["synonyms", "format", "edge_type", "joins", "source"])
def test_unsupported_field_level_keys_raise(key):
    def mutate(document):
        document["dimensions"][1][key] = "anything"

    with pytest.raises(ConversionError, match=f"{key!r} is not supported yet"):
        _rules(mutate)


def test_an_expression_referencing_an_undeclared_table_raises():
    def mutate(document):
        document["dimensions"][1]["expr"] = (
            "EXISTS (SELECT 1 FROM gold_other other WHERE other.sk_plan = source.sk_plan)"
        )

    with pytest.raises(ConversionError, match="undeclared table.*gold_other"):
        _rules(mutate)


def test_duplicate_field_names_raise():
    def mutate(document):
        document["dimensions"].append(dict(document["dimensions"][1]))

    with pytest.raises(ConversionError, match="duplicate field name"):
        _rules(mutate)


@pytest.mark.parametrize("layout,reason", [
    ("both", "only one of"),
    ("neither", "'fields' or 'dimensions'"),
])
def test_exactly_one_field_collection_is_required(layout, reason):
    def mutate(document):
        if layout == "both":
            document["fields"] = document["dimensions"]
        else:
            del document["dimensions"]

    with pytest.raises(ConversionError, match=reason):
        _rules(mutate)


def test_a_missing_source_raises():
    with pytest.raises(ConversionError, match="rules YAML must declare a source"):
        _rules(lambda document: document.pop("source"))


def test_a_field_without_an_expr_raises():
    def mutate(document):
        del document["dimensions"][1]["expr"]

    with pytest.raises(ConversionError, match="must declare an expr"):
        _rules(mutate)


@pytest.mark.parametrize("anchor_count", [0, 2])
def test_rules_need_exactly_one_anchor(anchor_count):
    def mutate(document):
        if anchor_count == 0:
            document["dimensions"][0]["tags"].remove("rules_engine:anchor")
        else:
            document["dimensions"][1]["tags"].append("rules_engine:anchor")

    with pytest.raises(ConversionError, match="exactly one anchor dimension"):
        _rules(mutate)


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
