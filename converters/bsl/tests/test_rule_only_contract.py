"""The converter chains rule models onto already-built semantic evidence."""

import ibis
import pytest
from boring_semantic_layer import to_semantic_table

import ossie_bsl
import ossie_rules_engine
from ossie_bsl import convert_ossie_to_bsl
from ossie_bsl.internals import ConversionError


@pytest.fixture
def source_model():
    con = ibis.duckdb.connect()
    con.create_table("evidence", schema={"id": "string"})
    return to_semantic_table(con.table("evidence")).with_dimensions(id=lambda t: t.id)


@pytest.fixture
def rule_yaml():
    return """semantic_model:
- name: rules
  datasets:
  - name: rules
    source: evidence
    fields:
    - name: rule_id
      expression:
        dialects:
        - dialect: DATABRICKS
          expression: source.id
"""


def test_evidence_conversion_and_identity_are_absent_from_installed_packages():
    assert not hasattr(ossie_rules_engine, "convert_evidence_to_ossie")
    assert not hasattr(ossie_bsl, "convert_evidence_to_ossie")
    assert not hasattr(ossie_bsl, "with_node_identity")
    with pytest.raises(ImportError):
        from ossie_rules_engine import evidence
    with pytest.raises(ImportError):
        from ossie_bsl import identity


def test_physical_table_binding_is_not_a_converter_argument(rule_yaml, source_model):
    with pytest.raises(TypeError, match="tables"):
        convert_ossie_to_bsl(rule_yaml, source_model=source_model, tables={})


def test_source_model_is_required(rule_yaml):
    with pytest.raises(TypeError, match="source_model"):
        convert_ossie_to_bsl(rule_yaml)


@pytest.mark.parametrize("invalid", [None, object()])
def test_source_model_must_be_a_semantic_model(rule_yaml, invalid):
    with pytest.raises(ConversionError, match="source_model must be a SemanticModel"):
        convert_ossie_to_bsl(rule_yaml, source_model=invalid)


def test_relationships_cannot_be_chained(rule_yaml, source_model):
    import yaml

    document = yaml.safe_load(rule_yaml)
    document["semantic_model"][0]["relationships"] = [{
        "name": "joined_evidence",
        "from": "rules",
        "to": "rules",
        "from_columns": ["id"],
        "to_columns": ["id"],
    }]
    with pytest.raises(ConversionError, match="relationships"):
        convert_ossie_to_bsl(yaml.safe_dump(document), source_model=source_model)
