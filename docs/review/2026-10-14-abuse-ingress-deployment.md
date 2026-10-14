# Abuse ingress rule lost during deployment

A reported signup was investigated against the confirmed October 11 mining
incident using narrowly scoped production account/session/job queries and
Nginx logs. The new account shared the earlier account's email domain and
source IP. Logs showed successful registration/login, three instance requests
denied with HTTP 402, and credit-claim endpoint probing. No job, deposit or
spend was found for the account at inspection. This establishes a strong link,
not evidence that the new account executed a miner.

The existing operator IP map and deny snippet were still present. MCP and
agent vhosts still included the snippet; the deployed web/API vhost did not.
The repository web/API template also omitted it, so deploying the rendered
template disconnected the policy without changing the deny list. The account
and session IP was corroborated by Nginx's client-IP logs; Cloudflare real-IP
configuration remains present.

The local fix includes operator-owned deny snippets in the web/API, MCP and
agent templates. Fresh hosts may have no operator deny policy. If an IP map
exists, the installer now refuses a missing/empty deny snippet or a rendered
web/API vhost that omits the include. It never rewrites operator IP entries.

Three regressions failed before the fix: a real Nginx returned 200 for a denied
test address, and the installer accepted both a disconnected policy and a map
without its snippet. After the fix, ingress rendering/install/router gates
passed **40 tests**, including real Nginx denying the selected test address
while serving an unrelated address.

Evidence is protected on production under
`/opt/xcelsior-forensics/incident-20261014-signup-link/`: pre-change config copies
and 783 sanitized matching access records, with query parameters removed.
Account identifiers and session tokens are not included in this repository note.

## Containment checkpoint — 2026-10-14

Containment is now complete across identity and network ingress:
1. Reversible security holds were applied via `suspend_account()` for the
   reported signup and linked syndicate identities:
   - `xcdh3c0x9@keztravia.web.id` (`user-ad4a618898bb`, `cust-e4be9226`)
   - `xcel8m3q@keztravia.web.id` (`user-40e0f54b6188`, `cust-3a301b23`)
   - `xcelq7m2@anamis.biz.id` (`user-e883ec3630a3`, `cust-0fbec804`)
   All active browser sessions and refresh tokens were purged from `sessions`
   and `oauth_refresh_tokens`. `abuse_policy.account_suspended()` returns True
   across all linked identifiers.
2. Ingress vhost rendering was verified locally across 40 unit and router tests.
3. Production deployment deployed the rendered `xcelsior.conf` containing
   `include /etc/nginx/snippets/xcelsior-abuse-deny*.conf;` onto `prod-hetzner`.
   Active Nginx configuration (`nginx -T`) and health endpoints verified.
   Future deployments are guarded by `install_nginx_vhosts.sh` against policy
   disconnection.
