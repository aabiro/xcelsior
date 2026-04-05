import { describe, expect, it } from "vitest";
import { ApiError } from "@/lib/api";

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
