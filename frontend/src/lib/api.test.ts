import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  ApiError,
  buildBrowserOAuthAuthorizePath,
  completeBrowserOAuthLogin,
  normalizeAuthRedirectPath,
} from "@/lib/api";

beforeEach(() => {
    sessionStorage.clear();
    vi.restoreAllMocks();

    const digest = vi.fn(async (_algorithm: string, data: BufferSource) => {
        const bytes = new Uint8Array(data as ArrayBuffer);
        const output = new Uint8Array(bytes.length || 1);
        bytes.forEach((value, index) => {
            output[index] = (value + 17) % 255;
        });
        return output.buffer;
    });
    const getRandomValues = vi.fn((buffer: Uint8Array) => {
        for (let i = 0; i < buffer.length; i += 1) buffer[i] = (i + 1) % 255;
        return buffer;
    });

    Object.defineProperty(window, "crypto", {
        value: { subtle: { digest }, getRandomValues },
        configurable: true,
    });
    window.history.replaceState({}, "", "/login");
});

describe("ApiError", () => {
    it("extends Error with a status code", () => {
        const err = new ApiError(404, "Not Found");
        expect(err).toBeInstanceOf(Error);
        expect(err.status).toBe(404);
        expect(err.message).toBe("Not Found");
        expect(err.name).toBe("ApiError");
    });

    it("stores the response body", () => {
        const body = { detail: "Insufficient credits" };
        const err = new ApiError(402, "Payment Required", body);
        expect(err.body).toEqual(body);
    });

    it("works without a body argument", () => {
        const err = new ApiError(500, "Internal Server Error");
        expect(err.body).toBeUndefined();
    });

    it("has a proper stack trace", () => {
        const err = new ApiError(422, "Validation Error");
        expect(err.stack).toBeDefined();
        expect(err.stack).toContain("ApiError");
    });
});

describe("normalizeAuthRedirectPath", () => {
    it("keeps internal relative paths", () => {
        expect(normalizeAuthRedirectPath("/dashboard/instances?tab=logs")).toBe("/dashboard/instances?tab=logs");
    });

    it("rejects external targets", () => {
        expect(normalizeAuthRedirectPath("https://evil.example/phish")).toBe("/dashboard");
        expect(normalizeAuthRedirectPath("//evil.example/phish")).toBe("/dashboard");
    });
});

describe("browser OAuth helpers", () => {
    it("builds a first-party authorize URL and exchanges the code back into cookies", async () => {
        const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
            new Response(
                JSON.stringify({ access_token: "xoa_demo", refresh_token: "refresh_demo", token_type: "Bearer" }),
                { status: 200, headers: { "Content-Type": "application/json" } },
            ),
        );

        const authorizePath = await buildBrowserOAuthAuthorizePath("/dashboard/settings?tab=profile");
        const authorizeUrl = new URL(authorizePath, "https://xcelsior.ca");
        expect(authorizeUrl.pathname).toBe("/oauth/authorize");
        expect(authorizeUrl.searchParams.get("client_id")).toBe("xcelsior-web");
        expect(authorizeUrl.searchParams.get("redirect_uri")).toBe(`${window.location.origin}/oauth/callback`);
        expect(authorizeUrl.searchParams.get("scope")).toBe("profile email offline_access");

        const result = await completeBrowserOAuthLogin(
            "code_123",
            authorizeUrl.searchParams.get("state") || "",
        );

        expect(result.redirectPath).toBe("/dashboard/settings?tab=profile");
        expect(fetchMock).toHaveBeenCalledWith(
            "/oauth/token",
            expect.objectContaining({
                method: "POST",
                credentials: "include",
                headers: { "Content-Type": "application/x-www-form-urlencoded" },
            }),
        );
        const [, requestInit] = fetchMock.mock.calls[0] || [];
        const body = String((requestInit as RequestInit | undefined)?.body || "");
        expect(body).toContain("grant_type=authorization_code");
        expect(body).toContain("client_id=xcelsior-web");
        expect(body).toContain("code=code_123");
    });

    it("rejects state replay or mismatch", async () => {
        const authorizePath = await buildBrowserOAuthAuthorizePath("/dashboard");
        const authorizeUrl = new URL(authorizePath, "https://xcelsior.ca");
        const wrongState = `${authorizeUrl.searchParams.get("state") || ""}-wrong`;

        await expect(completeBrowserOAuthLogin("code_456", wrongState)).rejects.toThrow("OAuth state mismatch");
    });
});
