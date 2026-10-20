"""Use real local HTTP to verify delivery failure and acknowledgement boundaries."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import threading
import urllib.error
import urllib.request

import pytest

spec = importlib.util.spec_from_file_location('xcelsior_alert_webhook', Path(__file__).resolve().parents[1]/'api/scripts/alert_webhook.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)

PAYLOAD = {'status':'firing','alerts':[{'labels':{'alertname':'DeliveryTest','severity':'critical'},'annotations':{'summary':'Controlled delivery test'}}]}


@contextmanager
def server(handler):
    http = ThreadingHTTPServer(('127.0.0.1',0),handler)
    runner = threading.Thread(target=http.serve_forever,daemon=True)
    runner.start()
    try:
        yield f'http://127.0.0.1:{http.server_port}'
    finally:
        http.shutdown()
        http.server_close()
        runner.join(timeout=2)


@pytest.mark.parametrize('upstream_status,receipt,expected',[(200,{'id':'real-receipt'},200),(503,{},503),(200,{},503)])
def test_webhook_ack_requires_a_successful_upstream_receipt(upstream_status,receipt,expected):
    requests=[]
    class Sink(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append({'token':self.headers.get('Authorization'),'priority':self.headers.get('Priority'),'body':self.rfile.read(int(self.headers['Content-Length']))})
            self.send_response(upstream_status)
            self.end_headers()
            self.wfile.write(json.dumps(receipt).encode())
        def log_message(self,*args):
            pass
    with server(Sink) as sink, server(adapter.handler(sink+'/topic','test-only-token')) as webhook:
        req=urllib.request.Request(webhook+'/alert',data=json.dumps(PAYLOAD).encode(),headers={'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req,timeout=3) as response:
                status=response.status
                assert json.load(response)['receipt_id']=='real-receipt'
        except urllib.error.HTTPError as error:
            status=error.code
        assert status==expected
        assert requests[0]['token']=='Bearer test-only-token'
        assert requests[0]['priority']=='high'
        assert b'Controlled delivery test' in requests[0]['body']


@pytest.mark.parametrize('payload',[[],{}, {'alerts':[]},{'alerts':[None]},{'alerts':[{'labels':None}]}])
def test_invalid_notification_payload_is_rejected(payload):
    with pytest.raises(ValueError):
        adapter.notification(payload)


def test_resolved_notifications_use_low_priority():
    title,body,priority=adapter.notification({**PAYLOAD,'status':'resolved'})
    assert 'RESOLVED' in title
    assert priority=='low'
    assert 'Controlled delivery test' in body
