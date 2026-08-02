# MCP tool versioning and deprecation policy

> Published contract for `https://mcp.xcelsior.ca/mcp`. Enforced by
> [`mcp/tests/unit/surface.test.ts`](../mcp/tests/unit/surface.test.ts), which
> fails the build on a breaking change that did not bump a version.
> Adoption plan item X6.28, gate GX6.

An agent that works today should still work next quarter without anyone
watching a changelog. This document says exactly what we may change, how much
warning you get, and how you find out.

---

## 1. Where the version lives

Every tool carries its version in the MCP metadata the protocol already gives
us:

```json
"_meta": {
  "xcelsior/toolVersion": "2.0.0",
  "xcelsior/idempotency": "keyed",
  "xcelsior/tenantClass": "tenant"
}
```

Versions are per **tool**, not per server: `create_instance` at 2.1.0 can sit
beside `list_instances` at 2.0.0. A client that pins behaviour should read
`_meta["xcelsior/toolVersion"]` from `tools/list` and compare it to what it was
built against.

Semantics:

| Part | Changes when |
|---|---|
| **Major** | A caller that worked before now fails. Only with the notice period below. |
| **Minor** | Something is added that existing callers can ignore — a new optional input, a new output field, a new accepted scope. |
| **Patch** | Behaviour is corrected to match what the tool already promised. Wording, an error message, a bug fix. |

---

## 2. What counts as breaking

This is not a judgement call at review time — it is the list the CI check
enforces, and it is derived by diffing the committed surface snapshot
(`mcp/tool-surface.json`) against the code.

**Breaking:**

- A tool is removed from the published profile, or moved to the operator profile.
- An input is removed, changes type, or an existing optional input becomes required.
- A new **required** input is added.
- An enum stops accepting a value it used to accept.
- A tool stops accepting a scope it previously accepted (the gateway grants
  access on *any* required scope, so removing an alternative narrows access).
- An annotation changes — `readOnlyHint`, `destructiveHint`, `idempotentHint`,
  `openWorldHint`. These are promises about behaviour, and a model that trusted
  one is entitled to keep trusting it.
- An output schema is removed.

**Not breaking:**

- A new optional input, or a widened enum.
- A new field in the output. Treat outputs as open — do not assume the set of
  keys is closed.
- A reworded description. Descriptions are tuned continuously for model
  behaviour; they are prose, not contract.
- A tool accepting an additional scope.
- A new tool.

---

## 3. Notice period and overlap

| Change | Notice before it takes effect | Overlap |
|---|---|---|
| Removing a tool | **90 days** | The tool keeps working for the full period and returns a `deprecation` field in its output naming the replacement. |
| Breaking change to an existing tool | **90 days** | The old shape keeps working for the full period. Where both shapes cannot coexist, the new behaviour ships as a **new tool name** and the old one enters the removal path above. |
| Removing a scope from a tool's accepted set | **90 days** | Same as above. |
| Tightening a rate limit | **30 days** | The published limits page is updated when the notice starts. |
| Security fix that cannot wait | Immediate | Announced within 24 hours with the reason. This is the only exception, and it is not a general-purpose escape hatch: it applies when leaving the behaviour in place would expose customer data or allow an unauthorized action. |

The clock starts when the change is published in the changelog (§4), not when
it is decided internally.

---

## 4. How you find out

1. **`tools/list`** — the version is in `_meta`, always current. A client that
   compares it against a stored value needs nothing else.
2. **Changelog** — [`mcp/CHANGELOG.md`](../mcp/CHANGELOG.md), one entry per
   surface change, breaking changes first.
3. **Deprecation field** — a deprecated tool includes
   `deprecation: { replaced_by, removal_date }` in its structured output, so an
   agent using it sees the notice at call time rather than in a document.

---

## 5. What this policy does **not** cover

- **Data, not shape.** GPU availability, prices, and host reputation change
  constantly by design. That is the marketplace, not the contract.
- **Tool descriptions.** Continuously tuned. If your integration depends on the
  exact wording of a description, it depends on the wrong thing.
- **The operator profile.** `ops-mcp` is unlisted, internal, and versioned at
  our discretion. Nothing in this document applies to it.

---

## 6. For maintainers

The snapshot is the contract. To change the surface:

```bash
cd mcp
# 1. Make the change.
# 2. If it is breaking, bump that tool's version in src/tools/contracts.ts and
#    add a CHANGELOG entry with the removal date.
npm run surface:update      # 3. Record the new shape
npm test                    # 4. The gate confirms the change is accounted for
```

`npm test` fails with the exact field that changed when a breaking change has
no version bump. Do not update the snapshot to make the failure go away — that
is precisely the move the check exists to catch.
