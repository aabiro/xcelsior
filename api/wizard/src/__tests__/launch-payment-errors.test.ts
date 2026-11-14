// A launch's 402 is classified by what the server said, over real HTTP.
//
// The API answers errors as {"ok":false,"error":{"code","message"}}. The
// client read `data.detail`, which is never set, so every 402 read
// "Insufficient balance — add funds", including "Unable to reserve wallet
// funds for launch" for a wallet that had money in it.

import http from "node:http";
import { describe, expect, it } from "vitest";
import { getPlatformRate, launchInstance, listAvailableGpus, serverMessage, spendableCad, WizardError } from "../api-client.js";
import { providerRateFor } from "../wizard-flow.js";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

function serve(status: number, body: unknown, seen: { path?: string; body?: Record<string, unknown> } = {}) {
    return new Promise<{ baseUrl: string; close: () => void }>((resolve) => {
        const server = http.createServer((req, res) => {
            let raw = "";
            req.on("data", (chunk) => { raw += chunk; });
            req.on("end", () => {
                seen.path = req.url;
                seen.body = raw ? JSON.parse(raw) : undefined;
                res.writeHead(status, { "Content-Type": "application/json" }).end(JSON.stringify(body));
            });
        });
        server.listen(0, "127.0.0.1", () => {
            const addr = server.address();
            if (!addr || typeof addr === "string") throw new Error("no port");
            resolve({ baseUrl: `http://127.0.0.1:${addr.port}`, close: () => server.close() });
        });
    });
}

const envelope = (message: string) => ({ ok: false, error: { code: "http_error", message } });

async function launchError(status: number, message: string): Promise<WizardError> {
    const { baseUrl, close } = await serve(status, envelope(message));
    try {
        await launchInstance(baseUrl, "tok", { name: "n", image: "img", gpu_model: "RTX 3060" });
    } catch (err) {
        return err as WizardError;
    } finally {
        close();
    }
    throw new Error("launch did not fail");
}

describe("launch 402s", () => {
    it("a shortfall keeps the server's numbers and says top up, then retry", async () => {
        const err = await launchError(402,
            "Insufficient available wallet balance for launch (~$0.35 CAD hold required; $0.10 CAD available after holds)");
        expect(err.code).toBe("insufficient_balance");
        expect(err.message).toContain("$0.10 CAD available");
        expect(err.remediation).toMatch(/top up/i);
        expect(err.remediation).toMatch(/retry/i);
        expect(err.url).toMatch(/\/dashboard\/billing$/);
    });

    it("a failed hold is shown as itself, not as a low balance", async () => {
        const err = await launchError(402, "Unable to reserve wallet funds for launch");
        expect(err.code).toBe("payment_required");
        expect(err.message).toBe("Unable to reserve wallet funds for launch");
        expect(`${err.message} ${err.remediation}`).not.toMatch(/insufficient|top up/i);
    });

    it("a hold that does not cover the tier quote is not a low balance either", async () => {
        const err = await launchError(402, "Wallet hold does not cover this service-tier quote");
        expect(err.code).toBe("payment_required");
    });

    it("a suspended wallet says so", async () => {
        const err = await launchError(402, "Wallet suspended — please add funds to resume service");
        expect(err.code).toBe("wallet_suspended");
    });

    it("other statuses carry the server's message", async () => {
        const err = await launchError(409, "Host h is offline and not accepting new instances");
        expect(err.message).toBe("Host h is offline and not accepting new instances");
    });
});

describe("launch request", () => {
    it("names the GPU model and sends no host id unless one is pinned", async () => {
        const seen: { path?: string; body?: Record<string, unknown> } = {};
        const { baseUrl, close } = await serve(200, { ok: true, instance: { job_id: "j", name: "n", status: "queued", host_id: "" } }, seen);
        try {
            await launchInstance(baseUrl, "tok", { name: "n", image: "img", gpu_model: "RTX 3060", vram_needed_gb: 12 });
        } finally {
            close();
        }
        expect(seen.path).toBe("/instance");
        expect(seen.body).toMatchObject({ gpu_model: "RTX 3060", vram_needed_gb: 12 });
        expect(seen.body).not.toHaveProperty("host_id");
    });
});

describe("live GPU feed", () => {
    it("reads /api/v2/gpu/available", async () => {
        const seen: { path?: string } = {};
        const gpus = [{ gpu_model: "RTX 3060", vram_gb: 12, region: "ca-on", province: "ON", count_available: 1, price_per_hour_cad: 0.35 }];
        const { baseUrl, close } = await serve(200, { ok: true, gpus, source: "host_gpu_devices" }, seen);
        try {
            expect(await listAvailableGpus(baseUrl, "tok")).toEqual(gpus);
        } finally {
            close();
        }
        expect(seen.path).toBe("/api/v2/gpu/available");
    });
});

/** A pricing endpoint that only knows the listed VRAM variants. */
function servePricing(variants: Record<number, number>, standard: number) {
    const paths: string[] = [];
    return new Promise<{ baseUrl: string; close: () => void; paths: string[] }>((resolve) => {
        const server = http.createServer((req, res) => {
            paths.push(req.url ?? "");
            const url = new URL(req.url ?? "", "http://x");
            const vram = url.searchParams.get("vram_gb");
            const rate = vram === null ? standard : variants[Number(vram)];
            if (rate === undefined) {
                res.writeHead(503).end(JSON.stringify(envelope("No Stripe Price for this GPU variant")));
                return;
            }
            res.writeHead(200).end(JSON.stringify({
                ok: true, gpu_model: url.searchParams.get("gpu_model"), vram_gb: vram ? Number(vram) : 8, base_rate_cad: rate,
            }));
        });
        server.listen(0, "127.0.0.1", () => {
            const addr = server.address();
            if (!addr || typeof addr === "string") throw new Error("no port");
            resolve({ baseUrl: `http://127.0.0.1:${addr.port}`, close: () => server.close(), paths });
        });
    });
}

describe("getPlatformRate", () => {
    it("prices the exact VRAM variant nvidia-smi reports, rounded", async () => {
        const { baseUrl, close, paths } = await servePricing({ 8: 0.1, 12: 0.12 }, 0.1);
        try {
            const rate = await getPlatformRate(baseUrl, "NVIDIA GeForce RTX 3060", 11.76);
            expect(rate).toMatchObject({ rate_per_hour: 0.12, vram_gb: 12, standard_variant: false });
        } finally {
            close();
        }
        expect(paths).toHaveLength(1);
        expect(paths[0]).toContain("/api/pricing/rates?");
        expect(paths[0]).toContain("vram_gb=12");
        expect(paths[0]).toContain("mode=on_demand");
    });

    it("falls back to the model's standard variant when that size has no price", async () => {
        const { baseUrl, close, paths } = await servePricing({ 8: 0.1 }, 0.1);
        try {
            const rate = await getPlatformRate(baseUrl, "RTX 3060", 10);
            expect(rate).toMatchObject({ rate_per_hour: 0.1, standard_variant: true });
        } finally {
            close();
        }
        expect(paths).toHaveLength(2);
        expect(paths[1]).not.toContain("vram_gb");
    });

    it("fails with the server's reason when the model has no price at all", async () => {
        const { baseUrl, close } = await servePricing({}, undefined as unknown as number);
        try {
            await expect(getPlatformRate(baseUrl, "Mystery GPU", 16)).rejects.toThrow("No Stripe Price for this GPU variant");
        } finally {
            close();
        }
    });
});

describe("providerRateFor", () => {
    it("recommended is the platform rate; competitive is 15% under it", () => {
        expect(providerRateFor("recommended", 0.49)).toBe(0.49);
        expect(providerRateFor("competitive", 0.49)).toBe(0.42);
        expect(providerRateFor("competitive", 0.01)).toBe(0.01); // never zero
    });
});

it("no wizard source reads the legacy listing table", () => {
    // `/marketplace/search` returns every listing row ever written, including
    // test listings for hosts that never existed (an H100 and a B200 at $0.20/hr).
    const dir = join(__dirname, "..");
    const offenders = readdirSync(dir, { recursive: true, encoding: "utf8" })
        .filter((name) => /\.(ts|tsx)$/.test(name) && !name.includes("__tests__"))
        .filter((name) => readFileSync(join(dir, name), "utf8")
            .split("\n").some((line) => line.includes("/marketplace/search") && !/^\s*(\*|\/\/)/.test(line)));
    expect(offenders).toEqual([]);
});

describe("helpers", () => {
    it("serverMessage reads the API's error envelope", () => {
        expect(serverMessage(envelope("nope"))).toBe("nope");
        expect(serverMessage({ detail: "legacy" })).toBe("legacy");
        expect(serverMessage({})).toBeNull();
        expect(serverMessage(undefined)).toBeNull();
    });

    it("spendableCad uses what a launch can spend (after holds), not the raw balance", () => {
        const wallet = { customer_id: "c", balance_cad: 5, available_cad: 0.2, total_deposited_cad: 0, total_spent_cad: 0, status: "active" };
        expect(spendableCad(wallet)).toBe(0.2);
        expect(spendableCad({ ...wallet, available_cad: undefined })).toBe(5);
    });
});
