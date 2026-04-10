import { afterEach, describe, expect, it } from "vitest";
import http from "node:http";

import { createOAuthClient } from "../api-client.ts";

function startJsonServer(
    handler: (req: http.IncomingMessage, body: string, res: http.ServerResponse) => void,
): Promise<{ baseUrl: string; close: () => Promise<void> }> {
    return new Promise((resolve) => {
        const server = http.createServer((req, res) => {
            let body = "";
            req.on("data", (chunk) => {
                body += chunk.toString();
            });
            req.on("end", () => handler(req, body, res));
        });
        server.listen(0, "127.0.0.1", () => {
            const address = server.address();
            if (!address || typeof address === "string") {
                throw new Error("Could not determine test server address");
            }
            resolve({
                baseUrl: `http://127.0.0.1:${address.port}`,
                close: () =>
                    new Promise((done, reject) => server.close((err) => (err ? reject(err) : done()))),
            });
        });
    });
}

describe("createOAuthClient", () => {
    let closer: (() => Promise<void>) | null = null;

    afterEach(async () => {
        if (closer) {
            await closer();
            closer = null;
        }
    });

    it("creates a confidential client_credentials client with bearer auth", async () => {
        const server = await startJsonServer((req, body, res) => {
            expect(req.method).toBe("POST");
            expect(req.url).toBe("/api/oauth/clients");
            expect(req.headers.authorization).toBe("Bearer session-user-token");
            const payload = JSON.parse(body);
            expect(payload.client_name).toBe("CLI Wizard Worker");
            expect(payload.client_type).toBe("confidential");
            expect(payload.grant_types).toEqual(["client_credentials"]);
            expect(payload.scopes).toEqual(["api"]);
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify({
                ok: true,
                client: {
                    client_id: "oauth_123",
                    client_secret: "secret_456",
                    client_name: payload.client_name,
                    client_type: payload.client_type,
                    grant_types: payload.grant_types,
                    scopes: payload.scopes,
                },
            }));
        });
        closer = server.close;

        const client = await createOAuthClient(server.baseUrl, "session-user-token", {
            client_name: "CLI Wizard Worker",
            client_type: "confidential",
            redirect_uris: [],
            grant_types: ["client_credentials"],
            scopes: ["api"],
        });

        expect(client.client_id).toBe("oauth_123");
        expect(client.client_secret).toBe("secret_456");
    });

    it("throws when oauth client creation fails", async () => {
        const server = await startJsonServer((_req, _body, res) => {
            res.writeHead(403, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ detail: "forbidden" }));
        });
        closer = server.close;

        await expect(
            createOAuthClient(server.baseUrl, "session-user-token", {
                client_name: "CLI Wizard Worker",
            }),
        ).rejects.toThrow("OAuth client creation failed: HTTP 403");
    });
});
