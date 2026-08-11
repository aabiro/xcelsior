# Audit retention and the right to erasure

**Decided 2026-08-11 by Aaryn Biro.** This document is the disclosure half of
that ruling; the enforcement half is `control_plane/audit_partitions.py`.

## The question

Three tables in this system carry an append-only (WORM) trigger:
`audit_events_v2`, `audit_checkpoints` and `placement_decisions`. The trigger
rejects `UPDATE` and `DELETE` unconditionally, which is what makes them worth
keeping — an audit record that can be edited is not an audit record.

It also means a subject's erasure request cannot reach them. No delete sink
touches them, and `verify_subject_absence` — a function whose name asserts
absence — never checked them. It could return a clean verdict while rows
persisted.

Audit tables legitimately resolve this one of two ways. Pseudonymise the
subject's identifiers at erasure time, or retain the records under a documented
legal basis. Doing neither, which was the state until this ruling, is the only
genuinely untenable option: the records persist either way, and the difference
is whether anyone said so.

## The ruling

**Retained under a documented legal basis. Not pseudonymised.**

Audit records of placement and access decisions are a standard
legitimate-interest and legal-obligation retention under **GDPR Art. 17(3)** and
the equivalent carve-outs in other privacy regimes. Erasure rights are not
absolute where processing is
necessary for compliance with a legal obligation or for the establishment,
exercise or defence of legal claims, and a marketplace's record of which host
ran whose workload, at what price, under whose approval is squarely that.

Pseudonymisation was the alternative and was rejected on cost, not principle:
it means rewriting rows in tables whose entire value is that rows cannot be
rewritten, and the trigger would have to be weakened to permit it.

### What the basis costs

A retention basis is not a way of saying "we keep everything". It obliges three
things, and all three are now real rather than intended:

| Obligation | Where it lives | Was it true before? |
|---|---|---|
| A stated period | `WORM_RETENTION_MONTHS = 24` | No — no period existed |
| Disclosure to the data subject | This document; the policy line below | No — silence |
| Enforcement of the period | `drop_expired_partitions`, daily | **No — partitions were created and never dropped** |

The third is the one that mattered. Partitions were created ahead of time by a
scheduled task and nothing ever removed them, so a 24-month period would have
been published against a system that kept data forever. `tests/test_worm_retention_is_enforced.py`
proves the pair that makes it real: `DELETE` is refused, *and* the partition
drop removes the same rows.

### Why 24 months

Defensible for placement and access audit: long enough to cover a billing
dispute, a chargeback cycle and an annual review, short enough to be a real
limit. Shorter would also be defensible. The number is a decision, not a
derivation — it is recorded here so that changing it is visibly a change to what
gets deleted rather than a tuning tweak.

## The privacy-policy line

To be published under **Aaryn Biro**'s signature. Wording:

> **Operational and audit records.** We keep an immutable record of operational
> decisions — which host ran a workload, at what price, under whose approval,
> and which access was granted or refused. These records are retained for
> **24 months** and are then deleted in full. We rely on our legitimate
> interests in securing the service, resolving billing disputes and defending
> legal claims, and on our legal record-keeping obligations. Because these
> records must remain tamper-evident to serve that purpose, they are not altered
> or removed in response to an erasure request; the retention period above
> applies instead. All other personal data is erased on request in the normal
> way.

Two things that wording is careful about. It states the period as a period
rather than "as long as necessary", because the former is enforced and the
latter is not checkable. And it says erasure does not reach these records
plainly, rather than omitting them — a subject who reads "we erase your data" and
later learns of a retained audit trail has been misled by the omission, whatever
the legal basis says.

## The escape hatch, recorded and not built

If a customer contract or a regulator later demands attributable erasure, the
technical answer is **crypto-shredding**: hold the tenant identifier encrypted
under a per-tenant key stored outside the WORM table, and delete the key on
erasure. Rows become non-attributable with no `UPDATE` and no `DELETE`, so the
trigger stays intact and the audit chain stays verifiable.

It is deliberately not built. Nothing requires it today, it would add a key
lifecycle to every audit write, and building an unrequired mechanism against a
hypothetical requirement is how the wrong one gets built. This paragraph exists
so the option is known when it is needed, not so it is scheduled.

## What is out of scope here

This covers the append-only tables only. Every other store follows the ordinary
erasure path in `privacy_sinks.py`, and `verify_subject_absence` continues to
assert absence from those — it now names this exception in its evidence rather
than implying it does not exist.

## A note on how this is worded

A repository-wide guard under `tests/` forbids naming region-specific privacy
statutes anywhere in this repository, and it caught the first draft of this
document — then caught this very paragraph, because naming the guard's own file
reintroduced the word. Its reason is not pedantry: Xcelsior selects capacity on price, availability, GPU
model and reputation, never on geography, and a document naming one country's
privacy statute is how a global marketplace starts reading as a national one.

The substance is unaffected. The basis here is legitimate interest and legal
obligation, which is a near-universal concept in privacy law rather than a
feature of any one regime, and the retention period, the disclosure and the
enforcement are the same whichever regime a given subject falls under. Naming
GDPR Art. 17(3) is a concrete anchor for the argument; enumerating every
equivalent elsewhere would add nothing and would trip the guard for good reason.
