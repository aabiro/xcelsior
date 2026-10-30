#!/usr/bin/env python3
"""Provision approved tier Prices through Stripe MCP and import exact amounts.

No customer, subscription, wallet, or payment is modified. Superseded Stripe
Prices remain available by ID for existing commitments. The stable lookup key
moves to the new immutable Price. Run --apply explicitly to provision Prices;
--import-dsn separately installs the verified snapshot into a migrated database.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path

import requests
from dotenv import dotenv_values

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent.parent if _HERE.parent.name == "api" else _HERE.parent
sys.path[:0] = [str(ROOT / "billing"), str(ROOT / "core"), str(ROOT)]
from service_tier_pricing import APPROVED_PRICE_FACTORS, POLICY_VERSION, stripe_amount_micros


class StripeMcp:
    def __init__(self, config: Path, server: str):
        cfg = json.loads(config.read_text())["mcpServers"][server]
        if cfg["url"].rstrip("/") != "https://mcp.stripe.com":
            raise ValueError("Expected the official Stripe MCP endpoint")
        env = {**dotenv_values(ROOT / ".env"), **os.environ}
        self.url = cfg["url"]
        self.session = requests.Session()
        headers = {
            key: re.sub(r"\$\{([^}]+)\}", lambda match: env.get(match[1], "") or "", value)
            for key, value in cfg.get("headers", {}).items()
        }
        self.session.headers.update(headers)
        self.session.headers.update(
            {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
        )
        self.ident = 0
        self.rpc(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "xcelsior-service-tier-catalog", "version": "1.0"},
            },
        )

    def rpc(self, method, params):
        self.ident += 1
        response = self.session.post(
            self.url,
            json={"jsonrpc": "2.0", "id": self.ident, "method": method, "params": params},
            timeout=45,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Stripe MCP returned HTTP {response.status_code}")
        if response.headers.get("Mcp-Session-Id"):
            self.session.headers["Mcp-Session-Id"] = response.headers["Mcp-Session-Id"]
        if "text/event-stream" in response.headers.get("Content-Type", ""):
            messages = [
                json.loads(line[5:].strip())
                for line in response.text.splitlines()
                if line.startswith("data:")
            ]
            result = next((item for item in messages if item.get("id") == self.ident), {})
        else:
            result = response.json()
        if "error" in result:
            raise RuntimeError(result["error"].get("message", "Stripe MCP error"))
        return result.get("result", {})

    def tool(self, name, arguments):
        result = self.rpc("tools/call", {"name": name, "arguments": arguments})
        if result.get("structuredContent") is not None:
            value = result["structuredContent"]
        else:
            value = json.loads(
                "\n".join(
                    block.get("text", "")
                    for block in result.get("content", [])
                    if block.get("type") == "text"
                )
            )
        if result.get("isError") or value.get("isError"):
            raise RuntimeError("Stripe MCP refused the operation; inspect the saved response")
        return value

    def read(self, operation, parameters):
        return self.tool(
            "stripe_api_read", {"stripe_api_operation_id": operation, "parameters": parameters}
        )


def price_plan(prices, manifest, *, include_queue_bids=False):
    by_lookup = {p["lookup_key"]: p for p in prices if p.get("lookup_key")}
    plan = []
    for variant in manifest["gpu_variants"]:
        for entry in variant["prices"]:
            if entry["tier"] != "standard":
                continue
            base = by_lookup.get(entry["lookup_key"])
            if base is None or base["product"] != variant["product_id"]:
                raise ValueError(
                    f"Standard Stripe Price missing or product mismatch: {entry['lookup_key']}"
                )
            stripe_amount_micros(base)
            if not base.get("livemode") or not base.get("active"):
                raise ValueError("Expected active live Stripe Prices")
            for tier, factor in APPROVED_PRICE_FACTORS.items():
                lookup = base["lookup_key"].replace(":standard:", f":{tier}:")
                amount = Decimal(base["unit_amount_decimal"]) * factor
                current = by_lookup.get(lookup)
                existing = bool(
                    current
                    and Decimal(current["unit_amount_decimal"]) == amount
                    and current["product"] == base["product"]
                )
                plan.append(
                    {
                        "variant": {
                            key: variant[key]
                            for key in ("gpu_model", "vram_gb", "form_factor", "high_frequency")
                        },
                        "mode": entry["mode"],
                        "tier": tier,
                        "base": base,
                        "lookup_key": lookup,
                        "amount_decimal": str(amount),
                        "existing": current if existing else None,
                    }
                )
            if include_queue_bids and entry["mode"] == "on_demand":
                prefix = base["lookup_key"].rsplit(":", 2)[0]
                lookup = f"{prefix}:queue-bid:all"
                current = by_lookup.get(lookup)
                plan.append(
                    {
                        "variant": {
                            key: variant[key]
                            for key in ("gpu_model", "vram_gb", "form_factor", "high_frequency")
                        },
                        "mode": "all",
                        "tier": "queue-bid",
                        "base": base,
                        "lookup_key": lookup,
                        "amount_decimal": "1",
                        "existing": (
                            current
                            if current
                            and current["product"] == base["product"]
                            and stripe_amount_micros(current) == 10000
                            else None
                        ),
                    }
                )
    return plan


def import_catalog(conn, catalog):
    """Import exact verified Stripe amounts; never derive runtime prices here."""
    from psycopg.rows import dict_row

    version = catalog["catalog_version"]
    rows = catalog["prices"]
    with conn.cursor(row_factory=dict_row) as cur:
        # Serialize imports without locking job/host/attempt runtime rows.
        cur.execute("SELECT pg_advisory_xact_lock(hashtext('stripe-compute-catalog'))")
        for row in rows:
            price = row["price"]
            micros = stripe_amount_micros(price)
            previous = cur.execute(
                "SELECT unit_amount_micros, stripe_product_id FROM stripe_compute_prices WHERE stripe_price_id=%s",
                (price["id"],),
            ).fetchone()
            if previous and (
                previous["unit_amount_micros"] != micros
                or previous["stripe_product_id"] != price["product"]
            ):
                raise ValueError("An immutable Stripe Price changed amount or product")
            cur.execute(
                "UPDATE stripe_compute_prices SET active=FALSE WHERE lookup_key=%s AND stripe_price_id<>%s",
                (price["lookup_key"], price["id"]),
            )
            cur.execute(
                """INSERT INTO stripe_compute_prices(stripe_price_id,stripe_product_id,lookup_key,unit_amount_micros,livemode,catalog_version)
                   VALUES(%s,%s,%s,%s,%s,%s) ON CONFLICT(stripe_price_id) DO UPDATE SET active=TRUE""",
                (
                    price["id"],
                    price["product"],
                    price["lookup_key"],
                    micros,
                    price["livemode"],
                    version,
                ),
            )
            if row["tier"] in ("gpu-exclusive", "queue-bid"):
                continue
            variant = row["variant"]
            cur.execute(
                """INSERT INTO gpu_pricing(gpu_model,vram_gb,form_factor,high_frequency,tier,pricing_mode,base_rate_micros,stripe_price_id,active)
                   VALUES(%s,%s,%s,%s,%s,%s,%s,%s,TRUE)
                   ON CONFLICT(gpu_model,vram_gb,form_factor,high_frequency,tier,pricing_mode)
                   DO UPDATE SET base_rate_micros=EXCLUDED.base_rate_micros,stripe_price_id=EXCLUDED.stripe_price_id,active=TRUE""",
                (
                    variant["gpu_model"],
                    variant["vram_gb"],
                    variant["form_factor"],
                    variant["high_frequency"],
                    row["tier"],
                    row["mode"],
                    micros,
                    price["id"],
                ),
            )


def update_manifest(manifest_path: Path, catalog_path: Path):
    manifest = json.loads(manifest_path.read_text())
    catalog = json.loads(catalog_path.read_text())
    manifest["service_tier_catalog"] = catalog
    for variant in manifest["gpu_variants"]:
        matching = [
            row
            for row in catalog["prices"]
            if all(
                row["variant"][key] == variant[key]
                for key in ("gpu_model", "vram_gb", "form_factor", "high_frequency")
            )
        ]

        def entry(row):
            price = row["price"]
            return {
                "price_id": price["id"],
                "lookup_key": price["lookup_key"],
                "tier": row["tier"],
                "mode": row["mode"],
                "rate_cad_per_hr": float(Decimal(price["unit_amount_decimal"]) / 100),
                "nickname": price["nickname"],
            }

        variant["prices"] = [
            entry(row) for row in matching if row["tier"] not in ("gpu-exclusive", "queue-bid")
        ]
        variant["gpu_exclusive_addon_prices"] = [
            entry(row) for row in matching if row["tier"] == "gpu-exclusive"
        ]
        variant["queue_bid_unit_prices"] = [
            entry(row) for row in matching if row["tier"] == "queue-bid"
        ]
    manifest["gpu_exclusive_price_rows"] = sum(
        row["tier"] == "gpu-exclusive" for row in catalog["prices"]
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcp-config", type=Path, default=Path.home() / ".cursor/mcp.json")
    parser.add_argument("--mcp-server", default="stripe-xcelsior")
    parser.add_argument("--manifest", type=Path, default=ROOT / "billing/stripe_catalog.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--include-queue-bids",
        action="store_true",
        help="Provision a CAD 0.01/GPU-hour bid unit Price per GPU variant",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--import-dsn-env",
        help="Import the saved catalog using this environment variable; never put a DSN on the command line",
    )
    parser.add_argument(
        "--update-manifest-from",
        type=Path,
        help="Publish a verified snapshot in the versioned deployment manifest, without network access",
    )
    args = parser.parse_args()
    if args.update_manifest_from:
        update_manifest(args.manifest, args.update_manifest_from)
        print("Updated versioned Stripe deployment manifest")
        return
    if args.import_dsn_env:
        import psycopg

        with psycopg.connect(os.environ[args.import_dsn_env]) as conn:
            data = json.loads(args.output.read_text())
            import_catalog(conn, data.get("service_tier_catalog", data))
        print("Imported verified Stripe Price snapshot")
        return
    client = StripeMcp(args.mcp_config, args.mcp_server)
    client.tool("stripe_api_details", {"stripe_api_operation_id": "GetPrices"})
    client.tool("stripe_api_details", {"stripe_api_operation_id": "PostPrices"})
    prices = []
    while True:
        params = {"limit": 100, "active": True}
        if prices:
            params["starting_after"] = prices[-1]["id"]
        page = client.read("GetPrices", params)
        prices.extend(page["data"])
        if not page["has_more"]:
            break
    manifest = json.loads(args.manifest.read_text())
    include_queue_bids = args.include_queue_bids or any(
        variant.get("queue_bid_unit_prices") for variant in manifest["gpu_variants"]
    )
    plan = price_plan(prices, manifest, include_queue_bids=include_queue_bids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(
        f"Stripe MCP plan: {len(plan)} tier/add-on Prices, {sum(item['existing'] is None for item in plan)} to create",
        flush=True,
    )
    if not args.apply:
        return
    local = threading.local()
    journal_lock = threading.Lock()
    journal = args.output.with_suffix(".created.jsonl")

    def create(item):
        if item["existing"]:
            return {**item, "price": item["existing"]}
        if not hasattr(local, "client"):
            local.client = StripeMcp(args.mcp_config, args.mcp_server)
        base = item["base"]
        parameters = {
            "product": base["product"],
            "currency": "cad",
            "unit_amount_decimal": item["amount_decimal"],
            "recurring": {
                key: value
                for key, value in base["recurring"].items()
                if key in ("interval", "interval_count", "usage_type", "meter")
                and value is not None
            },
            "tax_behavior": "exclusive",
            "lookup_key": item["lookup_key"],
            "transfer_lookup_key": True,
            "nickname": f"{item['tier']} · {item['mode']} · {POLICY_VERSION}",
            "metadata": {
                "xcelsior_service_tier": item["tier"],
                "standard_price_id": base["id"],
                "pricing_policy": POLICY_VERSION,
                "scope": "physical_host" if item["tier"] == "dedicated" else "allocated_gpus",
            },
        }
        result = local.client.tool(
            "stripe_api_write", {"stripe_api_operation_id": "PostPrices", "parameters": parameters}
        )
        if not result.get("id") or result.get("object") != "price":
            with journal_lock:
                args.output.with_suffix(".approval.json").write_text(json.dumps(result, indent=2))
            raise RuntimeError("Stripe MCP did not create a Price; inspect approval/error response")
        if stripe_amount_micros(result) != int(Decimal(item["amount_decimal"]) * 10_000):
            raise ValueError("Stripe returned an unexpected amount")
        with journal_lock, journal.open("a") as stream:
            stream.write(json.dumps(result) + "\n")
        return {**item, "price": result}

    results = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 8))) as pool:
        futures = [pool.submit(create, item) for item in plan]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 50 == 0:
                print(f"Verified {len(results)}/{len(plan)} tier/add-on Prices", flush=True)
    catalog_rows = {}
    for item in results:
        variant = item["variant"]
        components = [(item["tier"], item["price"])]
        if item["mode"] != "all":
            components.append(("standard", item["base"]))
        for tier, price in components:
            catalog_rows[price["id"]] = {
                "variant": variant,
                "mode": item["mode"],
                "tier": tier,
                "price": price,
            }
    rows = sorted(catalog_rows.values(), key=lambda row: row["price"]["lookup_key"])
    digest = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.write_text(
        json.dumps({"catalog_version": f"{POLICY_VERSION}:{digest}", "prices": rows}, indent=2)
        + "\n"
    )
    print(
        f"Saved {len(rows)} exact Stripe Prices and source identities to {args.output}", flush=True
    )


if __name__ == "__main__":
    main()
