"""Record which credential registered an SSH public key.

`routes/ssh.py` guarded key registration with `_require_user_grant`, which
refuses `client_credentials` outright. That is why `register_ssh_key` — the tool
Gate P2's journey depends on — could not work for an OAuth-connected agent, and
why "launch → wait → connect → run → terminate using only tool calls" was
unreachable for one of the two credential types.

Allowing machine credentials to register a key is a narrowing of that guard, so
it comes with a record of who did it. `user_ssh_keys` previously carried only
`email` and `user_id`: a key registered by an agent was indistinguishable from
one a human pasted into the dashboard, which makes both audit and revocation
guesswork.

Two columns:

* `registered_by_client_id` — the OAuth client that registered the key, `NULL`
  for an interactive registration. This is the binding: a key registered by a
  machine credential is attributable to that credential, and revoking the client
  identifies exactly which keys to remove.
* `registered_by_auth_type` — how it was registered (`session`,
  `client_credentials`, `agent_api_key`), so the dashboard can show a human
  which of their keys an agent added without joining against the OAuth tables.

Both nullable with no default: existing rows predate the distinction and must
not be retroactively labelled as anything. `NULL` means "registered before this
was recorded", which is the truth.

The index is partial because only machine-registered keys are ever looked up
this way — revoking a client asks "which keys did *this* client add", never
"which keys have no client".
"""

from alembic import op

revision = "099"
down_revision = "098"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE user_ssh_keys "
        "ADD COLUMN IF NOT EXISTS registered_by_client_id TEXT"
    )
    op.execute(
        "ALTER TABLE user_ssh_keys "
        "ADD COLUMN IF NOT EXISTS registered_by_auth_type TEXT"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_user_ssh_keys_registered_by_client "
        "ON user_ssh_keys (registered_by_client_id) "
        "WHERE registered_by_client_id IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_user_ssh_keys_registered_by_client")
    op.execute("ALTER TABLE user_ssh_keys DROP COLUMN IF EXISTS registered_by_auth_type")
    op.execute("ALTER TABLE user_ssh_keys DROP COLUMN IF EXISTS registered_by_client_id")
