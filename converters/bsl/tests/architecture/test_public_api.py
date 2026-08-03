"""Pin the intentionally small OSSIE-to-BSL pipeline API."""

import ossie_bsl


def test_public_api_is_the_five_pipeline_operations():
    assert set(ossie_bsl.__all__) == {
        "convert_evidence_to_ossie",
        "convert_rules_to_ossie",
        "convert_ossie_to_bsl",
        "with_node_identity",
        "evaluate_rules",
    }
