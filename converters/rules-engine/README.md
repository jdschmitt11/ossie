<!--
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
-->

# Rules Engine &rarr; Apache Ossie converter

Converts Rules Engine metric YAML — the authoring format used for *evidence*
artifacts and *rule family* artifacts — into Apache Ossie semantic models.

```python
from ossie_rules_engine import convert_evidence_to_ossie, convert_rules_to_ossie

evidence_ossie = convert_evidence_to_ossie(
    evidence_yaml,
    model_name="self_plan_evidence",
    evidence_id="self.plan_evidence",
)
rules_ossie = convert_rules_to_ossie(rules_yaml, model_name="self_plan")
```

## Mapping

| Rules Engine metric YAML | Ossie |
| --- | --- |
| `source` | `dataset.source`, verbatim |
| `fields:` / `dimensions:` | `dataset.fields` (the two spellings are the same construct) |
| field `expr` | `field.expression.dialects[DATABRICKS]` |
| field `display_name` | `field.label` |
| field `comment` | `field.description`, verbatim |
| model `comment` | `semantic_model.description`, verbatim |
| the `rules_engine:anchor` field | `dataset.primary_key` |
| field `tags` | `field.custom_extensions[RULES_ENGINE]` |
| artifact and family identity | `semantic_model.custom_extensions[RULES_ENGINE]` |

Authored expressions use `TRY_CAST`, which is not ANSI SQL, so they are emitted
under the `DATABRICKS` dialect rather than `ANSI_SQL`.

## Extensions

Rules Engine metadata is carried in `custom_extensions` entries with
`vendor_name: RULES_ENGINE`. As the core spec requires, `data` is a JSON
*string*; each payload carries a `_v` schema marker.

Tags are parsed once into structured keys rather than stashed verbatim.
A tag the converter does not recognise raises `ConversionError` instead of
being silently dropped.

## Direction

Forward only (Rules Engine &rarr; Ossie). There is no reverse converter and no
CLI yet; both are additive and can be introduced when a consumer needs them.

## Scope

Supported today: a single dataset with `source`, `fields`/`dimensions`, `expr`,
`display_name`, `comment` and the tag vocabulary listed above.

Not yet supported — these raise `ConversionError` rather than being dropped:
`joins`, `measures`, `filter`, field-level `synonyms`, `format` and `edge_type`,
and the rule tags `rule_status_encoding`, `rule_threshold_ref` and
`rule_observation_roles`.

Every field expression is validated as Databricks SQL. In a single-source
artifact it may not reference an undeclared table; cross-table dependencies
must be represented by a future explicit multi-source construct rather than
implicitly resolving from the execution connection.

A rule's `rule_id` is an independent identity and need not equal its dimension
name (e.g. `MS-007`).
