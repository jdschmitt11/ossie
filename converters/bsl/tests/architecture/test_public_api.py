"""Pin the intentionally small OSSIE-to-BSL pipeline API."""

import ossie_bsl


def test_public_api_is_the_three_rule_pipeline_operations():
    assert set(ossie_bsl.__all__) == {
        "convert_rules_to_ossie",
        "convert_ossie_to_bsl",
        "evaluate_rules",
    }
