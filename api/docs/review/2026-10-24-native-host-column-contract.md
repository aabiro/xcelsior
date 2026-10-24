# Native host column and schema minimum contract

General host upserts now persist an explicitly supplied provider ID and preserve
an already recorded provider. No provider identity is inferred for old rows.
Host read projections use canonical region, country, province, capabilities and
administrative/availability fields alongside canonical ownership; conflicting
JSONB aliases cannot hide those facts.

An operator schema-minimum override may tighten the runtime requirement but
cannot lower it. Transition images keep their own deliberately qualified
minimum; an environment variable cannot make a newer incompatible writer serve
an older schema. Unknown/older configured minima fail closed.

The focused isolated PostgreSQL gate passed 79 tests, and the expanded impacted
writer/admission/schema selection passed 256 tests, each with zero failures,
errors or skips. Runnable gates include `tests/test_native_host_column_contract.py`,
`tests/test_population_writer_contracts.py`, `tests/test_host_fleet_disclosure_http.py`,
`tests/test_control_plane_schema.py` and
`tests/test_schema_floor_covers_what_the_code_uses.py` through the security harness.
This is a follow-up source candidate, separate from the frozen fdaab814 release.
Fleet placement still needs its remaining JSONB observation reads converted and
qualified; no complete JSONB-removal or production deployment claim is made.
