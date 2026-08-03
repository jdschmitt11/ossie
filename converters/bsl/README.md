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

# Apache Ossie &rarr; Boring Semantic Layer converter

Builds a lazy [Boring Semantic Layer](https://github.com/boringdata/boring-semantic-layer)
`SemanticModel` from an Ossie semantic model. Nothing is executed; the result is
an Ibis expression tree that materialises only on `.query(...).execute()`.

```python
from ossie_bsl import convert_ossie_to_bsl

evidence = convert_ossie_to_bsl(
    evidence_ossie_yaml,
    tables={"gold_plan_with_sk": con.table("gold_plan_with_sk")},
)
predicates = convert_ossie_to_bsl(rules_ossie_yaml, source_model=evidence)
```

## How expressions are handled

BSL's own `from_config` evaluates dimension strings as Ibis *deferred*
expressions with only `_` in scope, so it cannot represent SQL such as
`TRY_CAST` or a multi-branch `CASE WHEN`. This converter therefore does not use
`from_config`. It builds one projection per dataset with
`Table.sql(..., dialect="databricks")`, letting Ibis transpile the authored SQL,
and then wraps the projection with the native `to_semantic_table(...)` API:

```sql
SELECT <expression> AS <field>, ... FROM source
```

The dataset source is exposed to that query under the alias `source`, matching
the qualifier Ossie expressions produced by the Rules Engine converter use.

## Chaining a model onto another model

Passing `source_model=` projects over an existing `SemanticModel` instead of a
physical table. The upstream model is aliased `rule_input` and re-aliased to
`source` inside the query:

```sql
SELECT source.*, <expression> AS <field> FROM rule_input AS source
```

Reusing the alias `source` for both stages would produce a duplicate CTE name.
Because `source.*` passes every upstream column through, a field name that
collides with an upstream physical column raises rather than shadowing it.

## Metadata

Ossie extensions are preserved in `Dimension.metadata` under a namespaced
`ossie` key, because BSL merges dimension metadata flat into its JSON
definition and an un-namespaced key such as `description` would clobber the
built-in field:

- `metadata["ossie"]["field_extensions"]` &mdash; the field's `custom_extensions`
- `metadata["ossie"]["model_extensions"]` &mdash; the semantic model's
  `custom_extensions`, mirrored onto the primary-key dimension, since BSL has no
  model-level metadata slot

Fields named in `dataset.primary_key` are declared with `is_entity=True`. The
model `description` is passed through to the `SemanticModel`.

## Scope

One semantic model with one dataset. Relationships, metrics and multi-dataset
models raise `ConversionError` rather than being dropped.
