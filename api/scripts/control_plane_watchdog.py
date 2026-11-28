"""Send an independent heartbeat only after the public health probes pass."""

import json
import os
import sys
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def https_url(value: str) -> str:
    url = urlsplit(value)
    if url.scheme != "https" or not url.hostname or url.username or url.password or url.fragment:
        raise ValueError("Expected an HTTPS URL")
    return value


def probe(opener, label: str, url: str) -> bool:
    request = Request(https_url(url), headers={"Cache-Control": "no-cache, no-store"})
    with opener.open(request, timeout=8) as response:
        if response.status != 200:
            return False
        body = json.loads(response.read(65_536))
        if not isinstance(body, dict):
            return False
        if label == "API":
            return body.get("status") == "ready" and body.get("ok") is True
        return body.get("status") == "pass"


def main() -> int:
    opener = build_opener(NoRedirect())
    try:
        heartbeat = https_url(os.environ["XCELSIOR_HEALTHCHECKS_PING_URL"]).rstrip("/")
        if urlsplit(heartbeat).query or not urlsplit(heartbeat).path:
            raise ValueError("Expected a Healthchecks ping path")
        checks = (
            ("API", https_url(os.environ["XCELSIOR_WATCHDOG_API_URL"])),
            ("Headscale", https_url(os.environ["XCELSIOR_WATCHDOG_HEADSCALE_URL"])),
        )
    except (KeyError, ValueError):
        print("Invalid Healthchecks ping configuration", file=sys.stderr)
        return 1
    failures = []
    for label, url in checks:
        try:
            if probe(opener, label, url):
                continue
        except (OSError, ValueError):
            pass
        failures.append(label)
    try:
        if failures:
            print(f"Public probes failed for {', '.join(failures)}", file=sys.stderr)
            request = Request(heartbeat + "/fail", data=", ".join(failures).encode())
        else:
            request = Request(heartbeat, data=b"API and Headscale healthy")
        with opener.open(request, timeout=8) as response:
            if response.status != 200:
                raise URLError("heartbeat rejected")
    except (OSError, ValueError):
        print("Could not deliver the Healthchecks heartbeat", file=sys.stderr)
        return 1
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
