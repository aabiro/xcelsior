"""Validation failures stay serializable and do not echo submitted secrets."""

import json

import pytest
from fastapi.testclient import TestClient
from api import app


@pytest.mark.parametrize("password", ["secret", float("nan"), float("inf")])
def test_invalid_registration_is_422_without_echoing_raw_input(password):
    with TestClient(app) as client:
        response = client.post(
            "/api/auth/register",
            content=json.dumps(
                {
                    "email": "validation-private@example.test",
                    "password": password,
                }
            ),
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == 422, response.text
    errors = response.json()["error"]["details"]
    assert errors
    assert all(set(error) <= {"type", "loc", "msg"} for error in errors)
    assert "validation-private@example.test" not in response.text
    assert '"input"' not in response.text
