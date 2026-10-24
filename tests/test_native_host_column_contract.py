"""Native host facts and the real runtime schema floor cannot be overridden."""

import uuid

import pytest
from psycopg.types.json import Jsonb

from db import DatabaseOps, _get_pg_pool
from control_plane.schema_compat import (
    REQUIRED_MIN_REVISION,
    SchemaIncompatibleError,
    configured_range,
)


def test_host_writes_record_explicit_provider_and_native_reads_ignore_payload_aliases():
    host = uuid.uuid4().hex
    with _get_pg_pool().connection() as conn:
        try:
            DatabaseOps.upsert_host(
                conn,
                dict(
                    host_id=host,
                    owner="owner",
                    provider_id="provider",
                    region="ca-on",
                    country="CA",
                    province="ON",
                    capabilities={"egress_isolation": True},
                ),
                backend="postgres",
            )
            DatabaseOps.upsert_host(
                conn,
                dict(host_id=host, owner="intruder", provider_id="forged", region="ca-on"),
                backend="postgres",
            )
            conn.execute(
                "UPDATE hosts SET payload=payload||%s WHERE host_id=%s",
                (
                    Jsonb(
                        dict(
                            owner="payload-owner",
                            provider_id="payload-provider",
                            region="forged",
                            capabilities={"egress_isolation": False},
                        )
                    ),
                    host,
                ),
            )
            actual = DatabaseOps.get_host(conn, host, backend="postgres")
            assert actual["owner_id"] == "owner" and actual["provider_id"] == "provider"
            assert (
                actual["region"] == "ca-on"
                and actual["country"] == "CA"
                and actual["province"] == "ON"
            )
            assert actual["capabilities"] == {"egress_isolation": True}
            assert (
                next(
                    h
                    for h in DatabaseOps.load_hosts(conn, backend="postgres")
                    if h["host_id"] == host
                )
                == actual
            )
        finally:
            conn.rollback()


@pytest.mark.parametrize("revision", ["057", "139", "140", "typo"])
def test_operator_configuration_cannot_reenable_an_incompatible_schema(monkeypatch, revision):
    monkeypatch.setenv("XCELSIOR_DB_SCHEMA_MIN_REVISION", revision)
    with pytest.raises(SchemaIncompatibleError, match="cannot lower"):
        configured_range()


def test_operator_can_keep_the_runtime_minimum(monkeypatch):
    monkeypatch.setenv("XCELSIOR_DB_SCHEMA_MIN_REVISION", REQUIRED_MIN_REVISION)
    assert configured_range()[0] == REQUIRED_MIN_REVISION
