#!/usr/bin/env python3
"""Loopback Alertmanager-to-ntfy adapter; failed delivery stays retryable."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
from typing import Any
import urllib.request

log = logging.getLogger('xcelsior.alert_webhook')
MAX_BODY_BYTES = 256 * 1024


def notification(payload: Any) -> tuple[str, str, str]:
    if not isinstance(payload, dict):
        raise ValueError('alert group must be an object')
    alerts = payload.get('alerts')
    if not isinstance(alerts, list) or not 1 <= len(alerts) <= 100:
        raise ValueError('alert group requires 1-100 alerts')
    lines = []
    critical = False
    for alert in alerts:
        if not isinstance(alert, dict):
            raise ValueError('alert must be an object')
        labels, annotations = alert.get('labels', {}), alert.get('annotations', {})
        if not isinstance(labels, dict) or not isinstance(annotations, dict):
            raise ValueError('invalid labels or annotations')
        critical = critical or labels.get('severity') == 'critical'
        name = str(labels.get('alertname', 'Xcelsior alert'))
        summary = str(annotations.get('summary') or name)
        description = str(annotations.get('description') or '')
        lines.append(f'{name}: {summary}')
        if description and description != summary:
            lines.append(description)
    status = str(payload.get('status', 'firing')).upper()
    title = f'[{status}] Xcelsior: {len(alerts)} alert(s)'
    priority = 'low' if status == 'RESOLVED' else 'high' if critical else 'default'
    body = '\n'.join(lines)[:12000]
    return title, body, priority


def deliver(payload: Any, url: str, token: str, *, timeout: float = 10) -> str:
    title, body, priority = notification(payload)
    request = urllib.request.Request(url, data=body.encode(), method='POST', headers={
        'Title': title,
        'Priority': priority,
        'Content-Type': 'text/plain; charset=utf-8',
        'User-Agent': 'Xcelsior-Alertmanager/1.0',
        **({'Authorization': 'Bearer '+token} if token else {}),
    })
    with urllib.request.urlopen(request, timeout=timeout) as response:
        receipt = json.load(response)
    if not isinstance(receipt, dict) or not isinstance(receipt.get('id'), str) or not receipt['id']:
        raise ValueError('notification transport returned no message receipt')
    return receipt['id']


def handler(url: str, token: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            # No payloads, destination URLs, Authorization headers or secrets.
            log.info('request completed')

        def do_GET(self):
            self.send_response(200 if self.path == '/healthz' else 404)
            self.end_headers()

        def do_POST(self):
            if self.path != '/alert':
                self.send_error(404)
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= MAX_BODY_BYTES:
                    self.send_error(413)
                    return
                payload = json.loads(self.rfile.read(length))
                notification(payload)
            except (ValueError, TypeError, UnicodeDecodeError):
                self.send_error(400)
                return
            try:
                receipt = deliver(payload, url, token)
            except Exception as exc:
                log.warning('notification delivery failed: %s', type(exc).__name__)
                self.send_error(503, 'notification delivery failed; retry required')
                return
            response = json.dumps({'accepted': True, 'receipt_id': receipt}).encode()
            log.info('notification accepted by upstream; receipt_id=%s', receipt)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(response)))
            self.end_headers()
            self.wfile.write(response)
    return Handler


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    url = os.environ.get('NTFY_URL', '').strip()
    if not url.startswith(('http://', 'https://')):
        raise SystemExit('NTFY_URL must name the existing notification endpoint')
    server = ThreadingHTTPServer(('127.0.0.1', 9103), handler(url, os.environ.get('NTFY_TOKEN', '')))
    server.serve_forever()


if __name__ == '__main__':
    main()
