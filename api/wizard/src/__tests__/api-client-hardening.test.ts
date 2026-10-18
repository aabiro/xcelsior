import { describe, expect, it } from "vitest";
import http from "node:http";
import { parseSseBuffer } from "../sse-parser.js";
import { pollDeviceToken, requestDeviceCode } from "../api-client.js";

function startJsonServer(
    handler: (req: http.IncomingMessage, res: http.ServerResponse) => void,
): Promise<{ baseUrl: string; close: () => void }> {
    return new Promise((resolve) => {
        const server = http.createServer(handler);
        server.listen(0, "127.0.0.1", () => {
            const addr = server.address();
            if (!addr || typeof addr === "string") throw new Error("no port");
            resolve({
                baseUrl: `http://127.0.0.1:${addr.port}`,
                close: () => server.close(),
            });
        });
    });
}

describe("sse-parser", () => {
    it("parses CRLF multi-line data frames", () => {
        const raw = "event: token\r\ndata: {\"type\":\"token\",\"content\":\"hi\"}\r\n\r\n";
        const { frames } = parseSseBuffer(raw);
        expect(frames).toHaveLength(1);
        expect(frames[0].event).toBe("token");
        expect(JSON.parse(frames[0].data).content).toBe("hi");
    });

    it("ignores heartbeat comments", () => {
        const raw = ": ping\n\ndata: {\"type\":\"done\"}\n\n";
        const { frames } = parseSseBuffer(raw);
        expect(frames).toHaveLength(1);
        expect(JSON.parse(frames[0].data).type).toBe("done");
    });
});

describe("device poll RFC8628", () => {
    it("handles pending then authorized", async () => {
        let calls = 0;
        const { baseUrl, close } = await startJsonServer((_req, res) => {
            calls += 1;
            if (calls === 1) {
                res.writeHead(428).end(JSON.stringify({ detail: "authorization_pending" }));
                return;
            }
            res.writeHead(200).end(JSON.stringify({ access_token: "tok", token_type: "bearer", expires_in: 3600 }));
        });
        try {
            const pending = await pollDeviceToken(baseUrl, "dev");
            expect(pending.status).toBe("pending");
            const ok = await pollDeviceToken(baseUrl, "dev");
            expect(ok.status).toBe("authorized");
            if (ok.status === "authorized") expect(ok.token.access_token).toBe("tok");
        } finally {
            close();
        }
    });

    it("returns slow_down on 429", async () => {
        const { baseUrl, close } = await startJsonServer((_req, res) => {
            res.writeHead(429).end(JSON.stringify({ detail: "slow_down" }));
        });
        try {
            const result = await pollDeviceToken(baseUrl, "dev");
            expect(result.status).toBe("slow_down");
        } finally {
            close();
        }
    });

    it("returns expired on 410", async () => {
        const { baseUrl, close } = await startJsonServer((_req, res) => {
            res.writeHead(410).end(JSON.stringify({ detail: "expired_token" }));
        });
        try {
            const result = await pollDeviceToken(baseUrl, "dev");
            expect(result.status).toBe("expired");
        } finally {
            close();
        }
    });
});

describe("json retries", () => {
    it("requestDeviceCode retries transient 503", async () => {
        let calls = 0;
        const { baseUrl, close } = await startJsonServer((_req, res) => {
            calls += 1;
            if (calls < 3) {
                res.writeHead(503).end("{}");
                return;
            }
            res.writeHead(200).end(JSON.stringify({
                device_code: "d",
                user_code: "U",
                verification_uri: "http://x/verify",
                expires_in: 600,
                interval: 5,
            }));
        });
        try {
            const result = await requestDeviceCode(baseUrl);
            expect(result.device_code).toBe("d");
            expect(calls).toBeGreaterThanOrEqual(3);
        } finally {
            close();
        }
    });
});