# Address autocomplete

Address search uses Google Places API (New), followed by HERE and Photon when
Google is unavailable or returns no predictions. Manual entry stays available.
Older clients without a session token continue using HERE and Photon.

The API reads `XCELSIOR_GOOGLE_PLACES_API_KEY`. It must stay server-only. The
browser calls Xcelsior's authenticated `/api/locations/search` and
`/api/locations/resolve` routes; it never receives the Google key. Both API
Compose services pass this variable. No Maps JavaScript API key is needed.

In the `xcelsior-gpu` GCP project, the dedicated keys are:

| Key ID | Application restriction | API restriction |
| --- | --- | --- |
| `xcelsior-places-server-prod` | `46.225.20.97`, `2a01:4f8:1c19:b508::1` | `places.googleapis.com` |
| `xcelsior-places-server-dev` | `66.222.185.166` (current development egress) | `places.googleapis.com` |

If the server, ISP or VPN changes its outbound address, update the matching
key's IP restriction. Website referrer restrictions do not work for this flow.
Use the Xcelsior principal and explicit project flags for GCP administration;
the workstation's default gcloud project may belong to another company.

The browser reuses a random token across keystrokes and resolves only the
selected prediction. Selection, manual entry, country changes and a three-minute
session lifetime cause a new token for subsequent searches. Place Details uses
the same token as the selected prediction and requests only
`addressComponents,formattedAddress`. Google responses are not put in the
server's HERE/Photon caches. A provider error causes a one-minute backoff.

With these Essentials fields, session tokens do **not** make every autocomplete
request free. Google's session pricing bills the first twelve requests, makes
requests thirteen onward free, and separately bills the terminating details
lookup, subject to each SKU's free allowance. See
[Google's session pricing](https://developers.google.com/maps/documentation/places/web-service/session-pricing).

Google results display Google Maps attribution. The English and French privacy
notices explain the information sent to providers and link Google's policy;
the terms link Google's Maps terms.

Verification includes API authentication and identifier validation, token reuse,
address parsing, fallback and backoff, frontend token rotation and stale-response
protection. Live credential checks must separately confirm a successful Google
lookup from the allowed server IP and rejection from an unapproved IP or API.

Compute Engine MCP OAuth credentials are separate from these Places keys.
Claude's connector uses a web OAuth client with
`https://claude.ai/api/mcp/auth_callback`; Antigravity's manual OAuth setup needs
`https://antigravity.google/oauth-callback`. VS Code's native MCP integration
uses the desktop client for loopback authentication. The server URL is
`https://compute.googleapis.com/mcp`. Each application still requires its own
interactive Google authorization before its connection can be considered tested.
