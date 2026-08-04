# Incident diagnostics

Read-only tools for answering *what is production actually doing*, as opposed to
what the repository says it should be doing. Both are safe to run against a live
deployment: neither writes, and neither prints a secret.

They live here rather than in a scratch directory because a script that
establishes what production is doing should not itself be unattributable — the
same argument that motivates `identify_deployed_files.sh` in the first place.

## `audit_operator_clients.py`

Lists OAuth clients holding a scope that confers platform-operator authority.
`control_plane_v1._require_host_operator` authorizes a machine principal on its
scope alone, so any client holding one can act as an operator with no role
check, and every write to `oauth_clients.scopes` is an authorization decision.

```
XCELSIOR_POSTGRES_DSN='postgresql://…' python3 scripts/incident/audit_operator_clients.py
```

Needs only `psycopg` — no repo import, so it runs on a production host without
dragging the application in. Exit 0 clean, 1 if any client holds one, 2 if it
could not run.

**Run it before applying a fix.** Patching changes what is observable: a row
created earlier still exists afterwards, but you lose the ability to distinguish
"never happened" from "happened, then was cleaned up" by re-testing the route.

Legitimate holders are system paths only — first-party seeded defaults
(`created_by_email` NULL, `is_first_party=1`) and system-managed rows. Anything
with a real `created_by_email` was minted by a user.

## `identify_deployed_files.sh`

Names the exact revision of individual deployed files.

`scripts/deploy.sh` rsyncs the deploying machine's **working tree**, not a git
ref. So `git -C /opt/xcelsior rev-parse HEAD` answers the wrong question — it
reports what the deployer had checked out, which says nothing about file
contents if that tree was dirty, and `/opt/xcelsior` may carry no `.git` at all.

Content is the only ground truth. Each deployed file is hashed with git's blob
hash and matched against every version of that path in history.

```
REMOTE=user@host DEPLOY_DIR=/opt/xcelsior ./scripts/incident/identify_deployed_files.sh
```

**A `NO MATCH` result is a finding, not a failure.** It means the deployed
content corresponds to no commit — the tree was dirty when it shipped — which
answers the attribution question definitively in the negative: production cannot
be described by a revision, and any question of the form "was fix X running on
date Y" has to be answered by reading the deployed file, not by reading history.

A match identifies *that file*. It does not establish that the whole deployment
sits at that commit; under rsync each file is independent.

The default file set is the four that currently have open questions attached:
`routes/terminal.py` (SSH host-key pinning window), `api.py` (the startup
validation swallow path), `security.py` (committed dev signing secret and
deterministic encryption key), and `routes/auth.py` (operator-scope refusal).

## Not here

The live scope-refusal probe. It is a working escalation attempt, and it lands
as `tests/live/test_scope_refusals_live.py` — the file
`.github/workflows/live-gates.yml` already references — once the fix is
deployed, rather than shipping a tool ahead of the patch it tests.
