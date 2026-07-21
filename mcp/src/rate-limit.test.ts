import { describe, it, expect, beforeEach } from "vitest";
import {
  checkRateLimit,
  loadRateLimitConfig,
  resetRateLimitStateForTests,
  setRedisClientForTests,
  type RedisLike,
} from "./rate-limit.js";

beforeEach(() => {
  resetRateLimitStateForTests();
});

describe("loadRateLimitConfig", () => {
  it("defaults to memory without redis url", () => {
    const cfg = loadRateLimitConfig({});
    expect(cfg.backend).toBe("memory");
  });

  it("selects redis when MCP_REDIS_URL is set", () => {
    const cfg = loadRateLimitConfig({
      MCP_REDIS_URL: "redis://localhost:6379/1",
    });
    expect(cfg.backend).toBe("redis");
    expect(cfg.failClosed).toBe(true);
  });

  it("require-redis forces redis backend and fail-closed", () => {
    const cfg = loadRateLimitConfig({
      MCP_RATE_LIMIT_REQUIRE_REDIS: "1",
      MCP_REDIS_URL: "redis://localhost:6379/2",
    });
    expect(cfg.backend).toBe("redis");
    expect(cfg.failClosed).toBe(true);
  });
});

describe("checkRateLimit memory", () => {
  it("allows under limit and 429s over limit", async () => {
    const cfg = {
      backend: "memory" as const,
      redisUrl: "",
      perMinute: 2,
      failClosed: false,
    };
    expect((await checkRateLimit("k", cfg)).ok).toBe(true);
    expect((await checkRateLimit("k", cfg)).ok).toBe(true);
    const over = await checkRateLimit("k", cfg);
    expect(over.ok).toBe(false);
    if (!over.ok) {
      expect(over.status).toBe(429);
      expect(over.code).toBe("rate_limit_exceeded");
    }
  });
});

describe("checkRateLimit redis", () => {
  it("uses redis incr and 429s over limit", async () => {
    let n = 0;
    const fake: RedisLike = {
      async incr() {
        n += 1;
        return n;
      },
      async pExpire() {
        return true;
      },
    };
    setRedisClientForTests(fake);
    const cfg = {
      backend: "redis" as const,
      redisUrl: "redis://test",
      perMinute: 2,
      failClosed: true,
    };
    expect((await checkRateLimit("a", cfg)).ok).toBe(true);
    expect((await checkRateLimit("a", cfg)).ok).toBe(true);
    const over = await checkRateLimit("a", cfg);
    expect(over.ok).toBe(false);
    if (!over.ok) expect(over.status).toBe(429);
  });

  it("fail-closed 503 when redis client missing and required", async () => {
    setRedisClientForTests(null);
    // Force init path to return null by using empty client injection
    resetRateLimitStateForTests();
    setRedisClientForTests(null);
    // getRedis will try real import; inject a throwing client path via set after
    // Simulate unavailable: set client that throws on incr
    setRedisClientForTests({
      async incr() {
        throw new Error("redis down");
      },
      async pExpire() {
        return true;
      },
    });
    const cfg = {
      backend: "redis" as const,
      redisUrl: "redis://down",
      perMinute: 10,
      failClosed: true,
    };
    const d = await checkRateLimit("x", cfg);
    expect(d.ok).toBe(false);
    if (!d.ok) {
      expect(d.status).toBe(503);
      expect(d.code).toBe("rate_limit_unavailable");
    }
  });

  it("does not unlimited-pass when failClosed and redis errors", async () => {
    setRedisClientForTests({
      async incr() {
        throw new Error("ECONNREFUSED");
      },
      async pExpire() {
        throw new Error("ECONNREFUSED");
      },
    });
    const denied = await checkRateLimit("z", {
      backend: "redis",
      redisUrl: "redis://x",
      perMinute: 1000,
      failClosed: true,
    });
    expect(denied.ok).toBe(false);
    if (!denied.ok) expect(denied.status).toBe(503);
  });
});
