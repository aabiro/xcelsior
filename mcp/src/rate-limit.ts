/**
 * Multi-replica-safe MCP rate limiting (blueprint §31).
 *
 * - backend=memory: process-local Map (dev only)
 * - backend=redis: shared INCR + EXPIRE; when required and Redis is down,
 *   fail closed (503) — never silently unlimited
 */

export type RateLimitBackend = "memory" | "redis";

export type RateLimitDecision =
  | { ok: true }
  | { ok: false; status: 429 | 503; code: string; message: string };

export type RateLimitConfig = {
  backend: RateLimitBackend;
  redisUrl: string;
  perMinute: number;
  /** When backend=redis and true, Redis errors become 503 not unlimited allow. */
  failClosed: boolean;
};

type MemoryBucket = { count: number; resetAt: number };

const memoryBuckets = new Map<string, MemoryBucket>();

/** Injected Redis client for tests; production uses dynamic import of `redis`. */
export type RedisLike = {
  incr(key: string): Promise<number>;
  pExpire(key: string, ms: number): Promise<unknown>;
  isOpen?: boolean;
  connect?: () => Promise<unknown>;
};

let _redis: RedisLike | null = null;
let _redisInit: Promise<RedisLike | null> | null = null;

export function resetRateLimitStateForTests(): void {
  memoryBuckets.clear();
  _redis = null;
  _redisInit = null;
}

export function setRedisClientForTests(client: RedisLike | null): void {
  _redis = client;
  _redisInit = Promise.resolve(client);
}

function memoryCheck(key: string, perMinute: number): RateLimitDecision {
  const now = Date.now();
  const windowMs = 60_000;
  let bucket = memoryBuckets.get(key);
  if (!bucket || now >= bucket.resetAt) {
    bucket = { count: 0, resetAt: now + windowMs };
    memoryBuckets.set(key, bucket);
  }
  bucket.count += 1;
  if (bucket.count > perMinute) {
    return {
      ok: false,
      status: 429,
      code: "rate_limit_exceeded",
      message: "Too many MCP requests; retry in 60s.",
    };
  }
  return { ok: true };
}

async function getRedis(url: string): Promise<RedisLike | null> {
  if (_redis) return _redis;
  if (_redisInit) return _redisInit;
  _redisInit = (async () => {
    try {
      // Dynamic import keeps typecheck working when node_modules has redis.
      const mod = await import("redis");
      const client = mod.createClient({ url });
      client.on("error", () => {
        /* logged by caller on command failure */
      });
      if (!client.isOpen) {
        await client.connect();
      }
      _redis = client as unknown as RedisLike;
      return _redis;
    } catch {
      return null;
    }
  })();
  return _redisInit;
}

async function redisCheck(
  key: string,
  cfg: RateLimitConfig,
): Promise<RateLimitDecision> {
  const client = await getRedis(cfg.redisUrl);
  if (!client) {
    if (cfg.failClosed) {
      return {
        ok: false,
        status: 503,
        code: "rate_limit_unavailable",
        message: "Rate limit backend unavailable; refusing unlimited MCP access.",
      };
    }
    // Explicit non-fail-closed: fall back to memory (dev only).
    return memoryCheck(key, cfg.perMinute);
  }
  try {
    const rkey = `mcp:rl:${key}`;
    const count = await client.incr(rkey);
    if (count === 1) {
      await client.pExpire(rkey, 60_000);
    }
    if (count > cfg.perMinute) {
      return {
        ok: false,
        status: 429,
        code: "rate_limit_exceeded",
        message: "Too many MCP requests; retry in 60s.",
      };
    }
    return { ok: true };
  } catch {
    if (cfg.failClosed) {
      return {
        ok: false,
        status: 503,
        code: "rate_limit_unavailable",
        message: "Rate limit backend unavailable; refusing unlimited MCP access.",
      };
    }
    return memoryCheck(key, cfg.perMinute);
  }
}

/** Pure decision entry used by HTTP handler and unit tests. */
export async function checkRateLimit(
  key: string,
  cfg: RateLimitConfig,
): Promise<RateLimitDecision> {
  if (cfg.perMinute <= 0) {
    // 0 or negative means "disabled" only for memory/dev; redis+failClosed still
    // requires backend health if operators set backend=redis.
    if (cfg.backend === "redis" && cfg.failClosed) {
      const client = await getRedis(cfg.redisUrl);
      if (!client) {
        return {
          ok: false,
          status: 503,
          code: "rate_limit_unavailable",
          message: "Rate limit backend unavailable; refusing unlimited MCP access.",
        };
      }
    }
    return { ok: true };
  }
  if (cfg.backend === "redis") {
    return redisCheck(key, cfg);
  }
  return memoryCheck(key, cfg.perMinute);
}

export function loadRateLimitConfig(env: NodeJS.ProcessEnv = process.env): RateLimitConfig {
  const backendRaw = (env.MCP_RATE_LIMIT_BACKEND || "").trim().toLowerCase();
  const redisUrl = (env.MCP_REDIS_URL || env.REDIS_URL || "").trim();
  // Production-shaped default: redis when URL present, else memory.
  let backend: RateLimitBackend =
    backendRaw === "redis" || backendRaw === "memory"
      ? (backendRaw as RateLimitBackend)
      : redisUrl
        ? "redis"
        : "memory";
  // Explicit require-redis: force redis backend and fail closed.
  const requireRedis =
    (env.MCP_RATE_LIMIT_REQUIRE_REDIS || "").toLowerCase() === "1" ||
    (env.MCP_RATE_LIMIT_REQUIRE_REDIS || "").toLowerCase() === "true" ||
    (env.MCP_RATE_LIMIT_REQUIRE_REDIS || "").toLowerCase() === "yes";
  if (requireRedis) {
    backend = "redis";
  }
  const failClosedEnv = (env.MCP_RATE_LIMIT_FAIL_CLOSED || "").toLowerCase();
  const failClosed =
    requireRedis ||
    failClosedEnv === "1" ||
    failClosedEnv === "true" ||
    failClosedEnv === "yes" ||
    // When operators chose redis backend, default fail-closed (blueprint §31).
    (backend === "redis" && failClosedEnv !== "0" && failClosedEnv !== "false");

  return {
    backend,
    redisUrl: redisUrl || "redis://127.0.0.1:6379/0",
    perMinute: Number(env.MCP_RATE_LIMIT_PER_MIN || "60"),
    failClosed,
  };
}
