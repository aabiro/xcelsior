# Promoting an artifact onto a volume

**Status:** plan, not implementation. Written 2026-08-08.
**Closes:** the last item of P3 in
[mcp-agent-native-implementation-plan.md](./mcp-agent-native-implementation-plan.md).
**Depends on:** the durable-state tools shipped in `8f16244` (volume CRUD,
attach, snapshot, and `get_artifact_expiry`).

> **Nothing here is built.** P0–P3's other work exposed capabilities that already
> existed; this one does not exist. Artifact storage and volumes are separate
> systems with no path between them, so the decisions below are about building
> that path rather than about publishing it.

---

## 1. Why it cannot be a thin wrapper

Every other tool in this plan wraps an endpoint. This one has no endpoint to
wrap, and the reason is architectural rather than an oversight:

| | **Artifacts** | **Volumes** |
|---|---|---|
| Storage | S3-compatible object store (B2 / R2 / local) | NFS export |
| Reached by | presigned HTTPS URL, TTL-bounded | `mount` on a host, into a container |
| Addressed by | `object_key` in a bucket | a path under a share |
| Lifetime | `retain_until` — a clock | none |
| Who can touch it | anything with the URL | a host on the storage network |

**Nothing in the API process can write to a volume.** The API server does not
mount the NFS export; the hosts do. So promotion is not a copy the backend can
perform — it is work that must happen on a machine that can see both the object
store and the share, which today means a **worker host running the agent**.

That single fact determines most of what follows.

---

## 2. The shape

```
tool          promote_artifact_to_volume(job_id, volume_id?, name?, confirm)
   │
API           POST /api/v2/volumes/{volume_id}/promotions
   │            • authorise artifact + volume (both, separately)
   │            • resolve the artifact manifest
   │            • take a retention hold
   │            • insert a promotion row keyed for idempotency
   │            • enqueue_agent_command(host, "promote_artifacts", {promotion_id})
   │
agent         • fetch the manifest from the API (not from the command args)
              • mount the volume if it is not already mounted
              • for each file: request a fresh presigned URL, stream, verify
                sha256, write, mark that file done
              • report completion; release the hold
```

`enqueue_agent_command` already exists, already has an allowlist, a 16 KB args
cap and a TTL. `promote_artifacts` joins the allowlist beside `mount_volume` and
`unmount_volume`, which the agent already knows how to perform.

---

## 3. Six decisions, and what each costs

### 3.1 The command carries an id, not the work

The obvious design puts the file list and presigned URLs in the command args.
Two reasons not to:

* **The 16 KB cap.** A checkpoint directory with a few hundred shards exceeds it,
  and paginating a command row is worse than not needing to.
* **Presigned URLs are credentials.** A command row is queued, logged, and
  readable by anyone with database access for as long as it is retained. Putting
  time-limited read grants for a tenant's weights in one is avoidable, so avoid
  it.

The agent receives `{promotion_id}` and fetches the manifest over its
authenticated channel, requesting each URL as it reaches that file.

### 3.2 Idempotency is a stored key, not a guess

Gate P3: *"Promotion is idempotent under retry; a repeated call produces one
volume, not two."*

The phrase "not two volumes" matters — it implies promotion may **create** the
volume when none is given, which is the natural agent flow ("save this
somewhere"). So the key must cover volume creation, not just the copy.

A `volume_promotions` row with a unique constraint on
`(tenant_id, job_id, idempotency_key)`, where the key defaults to a hash of the
resolved artifact set. A retry finds the existing row and returns its state.
`payment_intents` already uses exactly this shape — `ON CONFLICT DO NOTHING` plus
`rowcount` to distinguish new from replayed — and `charge_saved_card` surfaces
that as `replayed`. Promotion should say the same thing rather than silently
appearing to succeed twice.

### 3.3 The retention clock must be stopped, not raced

An artifact has `retain_until`, and a janitor deletes past it. A 40 GB
promotion that starts an hour before expiry loses its source mid-copy, and the
failure mode is a **partially written volume** the user believes is complete.

`storage.artifacts` already carries `legal_hold`. Promotion takes a hold on
every artifact in the manifest before the first byte moves, and releases it on
completion or abandonment. A hold that outlives its promotion is a leak, so the
release belongs in the same sweep that expires stale promotions — not only on
the success path.

**This is the clause most likely to be skipped and least likely to be noticed**,
because it only bites on artifacts near their expiry, which are exactly the ones
worth promoting.

### 3.4 Which host runs the copy

If the volume is attached to a running instance, that instance's host — it can
already see the mount.

If it is not attached, there is no obvious host, and this is the genuinely open
question. Three options, none free:

| | Cost |
|---|---|
| **Any healthy host** with the storage network | needs a placement rule; a busy GPU host now does I/O for someone else's promotion |
| **Mount on demand, then unmount** | the agent already has `mount_volume` / `unmount_volume`; adds a mount lifecycle to fail in |
| **A dedicated promotion worker** | clean, and a new deployable to operate |

Recommendation: **mount on demand on the least-loaded host with the volume's
region**, reusing the existing mount commands. It needs no new deployable and
no new network path. Revisit if promotions start competing with training for
host I/O — which is measurable before it is a problem.

### 3.5 Progress is per-file, because retries are certain

Weights are large and networks are not. A promotion that restarts from zero
after a failure at 38 GB will be retried by a human who then watches it fail
again.

Per-file rows with `done`/`bytes`/`sha256_verified`, so a resumed promotion
skips what is already verified. `sha256` and `size_bytes` are already on
`storage.artifacts`, so verification costs nothing extra to specify — and an
unverified copy is worse than no copy, because it looks like a backup.

### 3.6 The tool waits, or does not

`watch_instance` is the plan's waiting primitive and holds a poll for up to an
hour. Promotion of a large checkpoint can exceed that.

`promote_artifact_to_volume` should **return a `promotion_id` immediately** and
leave watching to a separate read, rather than blocking a tool call for an
unbounded time. The description must be explicit that the copy is still running
— an agent that reports "saved" when it means "started" is the failure this
whole phase exists to prevent.

---

## 4. Gate P3, clause by clause

| Clause | How it is met | How it is proven |
|---|---|---|
| Promotion is idempotent under retry; one volume, not two | unique key §3.2 | call twice, assert one `volume_promotions` row and one volume, second reports `replayed` |
| An artifact past `retain_until` is gone; a promoted volume is not | hold §3.3, volume has no clock | expire an artifact after promotion, assert the volume still reads |
| Round-trip: train → promote → mount in a *new* instance → read the weights, tool calls only | the tools in `8f16244` plus this one | scripted journey against staging |

The round-trip is the one that needs a staging environment, which does not exist
yet. That is a dependency worth stating now rather than discovering at the gate.

---

## 5. What would make this the wrong design

* **If volumes stop being NFS.** The whole "a host must run the copy" premise
  comes from the API not being able to write to the share. Object-store-backed
  volumes would make this a server-side copy and most of §3 evaporates.
* **If promotion is mostly small files.** The per-file resume, the hold, and the
  async handle all exist because checkpoints are large. For a few megabytes,
  they are ceremony.
* **If the janitor grows a promotion-aware skip.** Then §3.3's hold is
  duplicated logic, and one of the two will drift.

---

## 6. Sequence

Each step is independently useful, which matters because this competes with P4.

* **A0 — the manifest read.** `GET /api/v2/volumes/{id}/promotions/{pid}` and the
  agent-facing manifest. Nothing copies yet; the shape is reviewable.
* **A1 — one file, one host, no resume.** The narrowest end-to-end path: an
  attached volume, a single artifact, sha256 verified. Proves the boundary is
  crossable.
* **A2 — the hold and idempotency.** Gate P3's first two clauses.
* **A3 — many files, resume, unattached volumes.** The mount-on-demand path.
* **A4 — the tool.** `promote_artifact_to_volume`, async handle, description
  that distinguishes started from finished.

**A1 is the one that tells you whether this design survives contact.** If a host
cannot stream from the object store to the mount at a usable rate, everything
after it is rework, so it should be built before A2 rather than after.
