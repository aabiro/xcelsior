-- One-off cleanup: remove legacy `oauth_`-prefixed OAuth clients (pre `xoa_` rename).
-- Safe to run while there are no customers. It KEEPS:
--   * first-party clients (xcelsior-web, ...) — they are not `oauth_`-prefixed
--   * the new `xoa_` clients
-- and removes the legacy client rows plus their dependent refresh tokens and
-- agent sessions (there is no FK cascade, so all three tables are cleaned).
--
-- Run against the xcelsior Postgres on the prod host:
--   psql "$DATABASE_URL" -f scripts/cleanup_legacy_oauth_clients.sql
--
-- `~ '^oauth_'` matches a literal "oauth_" prefix (unlike LIKE, the underscore
-- is not a wildcard here).

BEGIN;

-- show what will be removed (appears in psql output)
SELECT c.client_id, c.client_name, u.email AS created_by_email
FROM oauth_clients c
LEFT JOIN users u ON u.user_id = c.created_by_user_id
WHERE c.client_id ~ '^oauth_';

DELETE FROM oauth_refresh_tokens WHERE client_id ~ '^oauth_';
DELETE FROM sessions             WHERE client_id ~ '^oauth_';
DELETE FROM oauth_clients        WHERE client_id ~ '^oauth_';

COMMIT;
