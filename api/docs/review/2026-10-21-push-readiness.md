# Control-plane push readiness — 2026-10-21

The push-preparation pass fixes the worker upgrade callback's inferred keyword
argument type, formats the changed Python files and regenerates the endpoint
inventory. All four previously missing operations are explicitly classified as
internal authentication or administrator surfaces, with reasons. The resulting
inventory contains 568 operations and 568 classifications.

Black's formatting preserved the parsed ASTs of all formatted files. The worker
callback now uses explicit keyword arguments, including its existing optional
before-restart callback, instead of mixing callable and string dictionary
values. Its Ed25519 signature was regenerated using the existing trust root;
all twelve distribution signatures verify.

The combined application and repository gate passed **816 tests**, zero
failures/errors/skips, on fresh isolated PostgreSQL with no Tower access or
live credentials. Three named tests were explicitly excluded: optional Compose
smoke, the future B16.2 contract-migration placement check, and a prose check
whose companion document is absent from the tracked checkout. These exclusions
are not counted as implementation or verification of those remaining gates.
Repository Ruff, changed-file Black and the full tracked Pyright gate passed.
Both generated-file checks passed, including the published OpenAPI documents.

A separate whole-tree Black check found **252 files byte-identical to
origin/main** that remain unformatted. This existing repository backlog is
outside the scoped cutover edits; the whole-tree format gate is not claimed
green. The six unrelated dirty workspace files remain preserved.

[Machine-readable evidence](2026-10-21-push-readiness-evidence.json) pins the
combined manifest and worker source hash. Push preparation does not constitute
a production rollout: the deployed application still uses verified runtime
`cb71106b`, and all repair policies remain `report_only`. Dedicated database
roles are the next deployment phase. Schema contraction/physical code removal,
durable fairness/preemption, repair qualification and historical-data
disposition retain their own gates. Apple Developer Program and CI cockpit
work remain outside scope.
