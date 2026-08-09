# Publishing a real SSH host-key fingerprint

*Gate P2 asks `open_instance_access` for "the SSH endpoint **plus the fingerprint
to verify**". It returns `host_key_fingerprint: null` today. This is how it stops
doing that.*

## 0. Directive

The same six prohibitions as `docs/artifact-promotion-plan.md` apply. Two more,
specific to this:

1. **Never return a fingerprint the platform did not observe.** A synthesised,
   defaulted, or inherited-from-a-sibling value is worse than `null`, because
   `null` makes a model say "this cannot be verified" and a wrong value makes it
   say "verified".
2. **Never widen the claim beyond what the mechanism supports.** This defends
   against a network attacker between the user and the gateway. It does not
   defend against a hostile provider host, and no wording in a tool result may
   imply otherwise.

## 1. What is actually true today

Three findings, each checked rather than assumed:

- **Every instance already has a unique host key.** `worker_agent.py` runs
  `ls /etc/ssh/ssh_host_*_key 2>/dev/null || ssh-keygen -A` inside the container
  during SSH setup. The keys exist; nothing reads them.
- **The gateway is a TCP relay, not an SSH terminator.** `ssh_port` is hardcoded
  to 22 container-side precisely because "sshd (listening on 22) would refuse the
  gateway's relay" (incident 2026-04-23), and the public port is DNAT'd
  (`scripts/xcelsior-iptables-fix.sh`). So **the key the container presents is the
  key the user's client sees.** If the gateway re-terminated SSH this whole plan
  would be publishing the wrong key, which is why it is the first thing verified.
- **`connect.xcelsior.ca` is one host for every instance.** Users therefore
  accumulate `known_hosts` entries keyed by `[connect.xcelsior.ca]:PORT`, and
  ports are reused across instances (`_compute_public_ssh_port(job_id)`), which
  is what produces the "REMOTE HOST IDENTIFICATION HAS CHANGED" warning users
  eventually hit. Publishing the fingerprint makes that warning *actionable*
  instead of alarming.

### 1a. What a real host corrected (2026-08-09)

A0 shipped as logging-only specifically to find out how much instance images
vary. Run against a live cloud GPU instance, it found two things this plan had
assumed rather than checked.

**The relay property is not universal.** The container ran **no sshd and held no
host keys at all**; the provider's proxy terminates SSH and pipes into the
container. The key a client verifies there belongs to the **proxy**, so reading
the container would publish the wrong fingerprint. This does not invalidate §1 —
Xcelsior's own gateway really is a DNAT relay, and that was verified — but it
means the feature is **host-class specific**, and any host reached through a
provider proxy must report `""`. It already does, which is the value of having
built the unknown case first.

**Ed25519-only was wrong.** The endpoint offered **RSA-2048 and no Ed25519**, so
the original reader would have said "unknown" for a host whose key was perfectly
verifiable. The reader now asks in the client's own negotiation order — Ed25519,
ECDSA, RSA — and takes the first that exists. Order matters rather than merely
coverage: on a dual-key host, publishing the RSA fingerprint when the client will
negotiate Ed25519 makes verification fail on a host that is fine, which trains
people to ignore the warning.

Both corrections came from one instance. That is the argument for A1 waiting on
fleet data rather than proceeding on this sample.

## 2. What this buys, stated exactly

SSH's default is trust-on-first-use: the first connection accepts whatever key
appears, and only later changes are detected. TOFU's weak moment is the first
connection, and that is the only connection an agent-launched instance ever has.

Delivering the fingerprint over the **authenticated HTTPS API** turns that first
connection into a verified one, because the fingerprint arrives out-of-band
relative to the SSH channel being verified.

**Its limit, which must be stated in the tool description and not only here.**
The fingerprint is reported *by the worker agent on the host*. A compromised host
can report the fingerprint of an sshd it controls, and the check will pass. So:

- ✅ Defends against a network attacker between user and gateway.
- ✅ Detects an instance whose identity changed under a reused port.
- ❌ Does **not** prove the host is honest. Trust in the fingerprint is exactly
  trust in the worker agent that reported it — no more.

## 3. Sequence

Risk order, smallest blast radius first. Each stage is independently revertible
and leaves the system honest if the next never ships.

### A0 — Read the fingerprint (no storage, no API, no tool)

Worker agent, immediately after the `ssh-keygen -A` step, reads what it just
ensured exists:

```
docker exec <container> ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
```

Parsed to the `SHA256:…` field. Ed25519 only at A0 — it is OpenSSH's default
`HostKeyAlgorithms` preference, so it is the key a modern client will actually
negotiate. RSA is added at A3 only if a real client is found that negotiates it;
publishing a key nobody verifies is noise that dilutes the one that matters.

**Ships alone, logging only.** One deploy where the value is observed and never
used, so that a parse that returns `""` on some base image is discovered before
anything depends on it. Instance images vary far more than the control plane
does; this is the stage that finds out how.

**Done when:** the fleet logs a plausible `SHA256:` for launched interactive
instances, and the rate of blanks is known rather than assumed.

### A1 — Report it

Added to the existing instance status callback. Nullable everywhere; an agent
that reports nothing is a normal state, not an error.

**Refusal test:** a worker reporting a malformed fingerprint (`"yes"`,
`"SHA256:"`, 4KB of junk) is rejected at the API boundary and stored as null.
The column is served to users and must never carry attacker-controlled text.

### A2 — Store it

One nullable column, expand-only per Alembic rule 5. It belongs to the
**attempt/container**, not the job: a requeue produces a new container with new
keys, so a fingerprint that outlives its container is a stale value that would
make a model report "verified" against the wrong host. It is cleared wherever
`_clear_job_output` is called, for the same reason and in the same place.

### A3 — Serve it

The route behind `open_instance_access` returns it, alongside the port it belongs
to. Fingerprint and port travel together or not at all — a fingerprint attached
to the wrong port is a failed verification that looks like an attack.

### A4 — Return it from the tool

`host_key_fingerprint` carries the real value when known. The `host_key_note`
becomes the verification instruction and, when the value is absent, keeps
today's honest wording. **The null path is not deleted** — older instances,
non-interactive launches, and agents that have not yet reported all legitimately
produce it, and it stays correct for them forever.

The user-facing command becomes checkable:

```
ssh-keyscan -p PORT connect.xcelsior.ca 2>/dev/null | ssh-keygen -lf -
```

compared against the published value before connecting.

## 4. Gate

- A launched instance's published fingerprint **matches what `ssh-keyscan`
  returns for its port**, asserted against a live instance. This is the whole
  claim; everything else is plumbing. A mock cannot establish it.
- A requeued instance publishes the **new** container's fingerprint, never the
  previous one. Asserted by requeueing and comparing.
- A malformed report is stored as null and never served.
- An instance with no reported fingerprint still returns `ok: true` with the
  honest note — the feature must not make access *harder* than before it existed.
- No fingerprint appears for an instance the caller cannot access. It is a weak
  identifier, but it is one, and the access rules do not change because a field
  is small.

## 5. What this plan does not promise

- **No `known_hosts` management.** Xcelsior publishes a value; it does not edit
  anyone's client configuration.
- **No provider attestation.** §2's limit is a property of the design, not a gap
  to be closed later by the same mechanism. Closing it needs hardware attestation
  or a platform-held CA signing host keys, which is a different project.
- **No retrofit.** Instances already running when A1 ships never report one and
  will show `null` until they are relaunched. Backfilling would mean reading keys
  from live containers on the fleet, which is a larger operation than the feature
  is worth.
