"""Native placement filters cannot recover geography or capabilities from JSONB."""

import uuid

import pytest
from psycopg.types.json import Jsonb

from control_plane.scheduler.filters import FilterContext, filter_hosts
from control_plane.scheduler.snapshot import read_fleet
from db import DatabaseOps, _get_pg_pool


@pytest.mark.parametrize("native_capability", [True, False])
def test_fleet_projection_uses_canonical_geography_and_capabilities(native_capability):
    host_id = "canonical-fleet-" + uuid.uuid4().hex
    with _get_pg_pool().connection() as conn:
        try:
            DatabaseOps.upsert_host(
                conn,
                {
                    "host_id": host_id,
                    "status": "active",
                    "region": "ca-on",
                    "country": "CA",
                    "province": "ON",
                    "capabilities": {"egress_isolation": native_capability},
                    "gpu_count": 1,
                    "gpu_model": "RTX 4090",
                    "total_vram_gb": 24,
                    "free_vram_gb": 24,
                },
                backend="postgres",
            )
            conn.execute(
                "UPDATE hosts SET admission_state='admitted',administrative_state='admitted',availability_state='ready',payload=payload||%s WHERE host_id=%s",
                (
                    Jsonb(
                        {
                            "region": "forged",
                            "country": "XX",
                            "province": "YY",
                            "capabilities": {"egress_isolation": not native_capability},
                        }
                    ),
                    host_id,
                ),
            )
            from tests._db_helpers import report_test_host_inventory

            report_test_host_inventory(host_id, conn=conn)
            host = next(h for h in read_fleet(conn).hosts if h["host_id"] == host_id)
            assert (host["region"], host["country"], host["province"]) == ("ca-on", "CA", "ON")
            assert host["capabilities"] == {"egress_isolation": native_capability}
            eligible, rejected = filter_hosts(
                {"num_gpus": 1, "region": "ca-on", "egress": "restricted"}, [host], FilterContext()
            )
            assert bool(eligible) == native_capability
            if not native_capability:
                assert "egress_isolation_unavailable" in {r.code for r in rejected[host_id]}
        finally:
            conn.rollback()


def test_blank_canonical_facts_cannot_fall_back_to_payload_claims():
    host_id = "blank-fleet-" + uuid.uuid4().hex
    with _get_pg_pool().connection() as conn:
        try:
            conn.execute(
                "INSERT INTO hosts(host_id,status,registered_at,payload,admission_state,administrative_state,availability_state, last_observed_at) VALUES(%s,'active',0,%s,'admitted','admitted','ready', clock_timestamp())",
                (
                    host_id,
                    Jsonb(
                        {
                            "region": "ca-on",
                            "country": "CA",
                            "province": "ON",
                            "gpu_count": 1,
                            "capabilities": {"egress_isolation": True},
                        }
                    ),
                ),
            )
            host = next(h for h in read_fleet(conn).hosts if h["host_id"] == host_id)
            assert host["region"] is None and host["country"] is None and host["province"] is None
            assert host["capabilities"] == {}
            eligible, rejected = filter_hosts({"region": "ca-on", "egress": "restricted"}, [host])
            assert eligible == []
            assert {r.code for r in rejected[host_id]} >= {
                "region_mismatch",
                "egress_isolation_unavailable",
            }
        finally:
            conn.rollback()
