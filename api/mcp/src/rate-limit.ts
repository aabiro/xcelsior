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
  launchPerHour?: number;
  serverlessPerMinute?: number;
  maxWatchesPerPrincipal?: number;
};

type MemoryBucket = { count: number; resetAt: number };

const memoryBuckets = new Map<string, MemoryBucket>();
const memoryConcurrent = new Map<string, number>();

/** Injected Redis client for tests; production uses dynamic import of `redis`. */
export type RedisLike = {
  incr(key: string): Promise<number>;
  pExpire(key: string, ms: number): Promise<unknown>;
  eval?(
    script: string,
    options: { keys: string[]; arguments: string[] },
  ): Promise<unknown>;
  decr?(key: string): Promise<number>;
  del?(key: string): Promise<number>;
  ping?(): Promise<string>;
  isOpen?: boolean;
  connect?: () => Promise<unknown>;
};

let _redis: RedisLike | null = null;
let _redisInit: Promise<RedisLike | null> | null = null;

const INCREMENT_WITH_TTL_LUA = `
local value = redis.call("INCR", KEYS[1])
if value == 1 then
  redis.call("PEXPIRE", KEYS[1], ARGV[1])
end
return value
`;

async function atomicIncrementWithTtl(
  client: RedisLike,
  key: string,
  windowMs: number,
): Promise<number> {
  if (client.eval) {
    const result = await client.eval(INCREMENT_WITH_TTL_LUA, {
      keys: [key],
      arguments: [String(windowMs)],
    });
    return Number(result);
  }
  // Minimal injected test doubles predate EVAL. Production's node-redis
  // client always takes the Lua path above.
  const count = await client.incr(key);
  if (count === 1) await client.pExpire(key, windowMs);
  return count;
}

function invalidateRedis(): void {
  _redis = null;
  _redisInit = null;
}

export function resetRateLimitStateForTests(): void {
  memoryBuckets.clear();
  memoryConcurrent.clear();
  _redis = null;
  _redisInit = null;
}

export function setRedisClientForTests(client: RedisLike | null): void {
  _redis = client;
  _redisInit = Promise.resolve(client);
}

function memoryCheck(key: string, perMinute: number): RateLimitDecision {
  return memoryWindowCheck(key, perMinute, 60_000);
}

function memoryWindowCheck(key: string, limit: number, windowMs: number): RateLimitDecision {
  const now = Date.now();
  let bucket = memoryBuckets.get(key);
  if (!bucket || now >= bucket.resetAt) {
    bucket = { count: 0, resetAt: now + windowMs };
    memoryBuckets.set(key, bucket);
  }
  bucket.count += 1;
  if (bucket.count > limit) {
    return {
      ok: false,
      status: 429,
      code: "rate_limit_exceeded",
      message: "Too many MCP requests; retry in 60s.",
    };
  }
  return { ok: true };
}

/** Multi-replica atomic counter for tool-specific traffic policy classes. */
export async function checkToolLimit(
  principalKey: string,
  clientId: string,
  tool: string,
  cfg: RateLimitConfig,
): Promise<RateLimitDecision> {
  let limit = cfg.perMinute;
  let windowMs = 60_000;
  let className = "tool";
  if (["create_instance", "create_serverless_endpoint"].includes(tool)) {
    limit = cfg.launchPerHour ?? 20; windowMs = 3_600_000; className = "launch";
  } else if (tool === "run_serverless_job") {
    limit = cfg.serverlessPerMinute ?? 120; className = "serverless";
  }
  if (limit <= 0) return { ok: true };
  const key = `${className}:${principalKey}:${clientId || "no-client"}:${tool}`;
  if (cfg.backend === "memory") return memoryWindowCheck(key, limit, windowMs);
  const client = await getRedis(cfg.redisUrl);
  if (!client) return cfg.failClosed
    ? { ok: false, status: 503, code: "rate_limit_unavailable", message: "Distributed tool policy unavailable." }
    : memoryWindowCheck(key, limit, windowMs);
  try {
    const count = await atomicIncrementWithTtl(client, `mcp:rl:${key}`, windowMs);
    return count <= limit
      ? { ok: true }
      : { ok: false, status: 429, code: "tool_limit_exceeded", message: `${className} limit exceeded.` };
  } catch {
    invalidateRedis();
    return cfg.failClosed
      ? { ok: false, status: 503, code: "rate_limit_unavailable", message: "Distributed tool policy unavailable." }
      : memoryWindowCheck(key, limit, windowMs);
  }
}

export async function acquireWatchSlot(
  principalKey: string,
  cfg: RateLimitConfig,
): Promise<{ decision: RateLimitDecision; release: () => Promise<void> }> {
  const limit = cfg.maxWatchesPerPrincipal ?? 3;
  const key = `mcp:watch:${principalKey}`;
  if (cfg.backend === "memory") {
    const count = (memoryConcurrent.get(key) ?? 0) + 1;
    if (count > limit) {
      return {
        decision: { ok: false, status: 429, code: "watch_concurrency_exceeded", message: "Too many concurrent watches." },
        release: async () => undefined,
      };
    }
    memoryConcurrent.set(key, count);
    return {
      decision: { ok: true },
      release: async () => {
        const remaining = Math.max(0, (memoryConcurrent.get(key) ?? 1) - 1);
        if (remaining) memoryConcurrent.set(key, remaining);
        else memoryConcurrent.delete(key);
      },
    };
  }
  const client = await getRedis(cfg.redisUrl);
  if (!client) return {
    decision: cfg.failClosed
      ? { ok: false, status: 503, code: "rate_limit_unavailable", message: "Watch concurrency policy unavailable." }
      : { ok: true },
    release: async () => undefined,
  };
  try {
    const count = await atomicIncrementWithTtl(client, key, 3_600_000);
    if (count > limit) {
      await client.decr?.(key);
      return {
        decision: { ok: false, status: 429, code: "watch_concurrency_exceeded", message: "Too many concurrent watches." },
        release: async () => undefined,
      };
    }
    return {
      decision: { ok: true },
      release: async () => { try { await client.decr?.(key); } catch { /* TTL is the crash backstop */ } },
    };
  } catch {
    invalidateRedis();
    return {
      decision: cfg.failClosed
        ? { ok: false, status: 503, code: "rate_limit_unavailable", message: "Watch concurrency policy unavailable." }
        : { ok: true },
      release: async () => undefined,
    };
  }
}

export async function recordAuthFailure(key: string, cfg: RateLimitConfig): Promise<RateLimitDecision> {
  const abuseCfg = { ...cfg, perMinute: 10 };
  if (abuseCfg.backend === "redis") return redisCheck(`abuse:${key}`, abuseCfg);
  return memoryCheck(`abuse:${key}`, 10);
}

export async function rateLimitReady(cfg: RateLimitConfig): Promise<boolean> {
  if (cfg.backend === "memory") return !cfg.failClosed;
  const client = await getRedis(cfg.redisUrl);
  if (!client) return false;
  try {
    return client.ping ? (await client.ping()) === "PONG" : true;
  } catch {
    invalidateRedis();
    return false;
  }
}

async function getRedis(url: string): Promise<RedisLike | null> {
  if (_redis && _redis.isOpen !== false) return _redis;
  if (_redis?.isOpen === false) {
    _redis = null;
    _redisInit = null;
  }
  if (_redisInit) return _redisInit;
  _redisInit = (async () => {
    try {
      // Dynamic import keeps typecheck working when node_modules has redis.
      const mod = await import("redis");
      const client = mod.createClient({
        url,
        socket: {
          connectTimeout: 1_000,
          // A limiter command must fail closed promptly. The next request
          // rebuilds the connection after recovery; it never waits inside an
          // unbounded background reconnect loop.
          reconnectStrategy: false,
        },
      });
      client.on("error", () => {
        /* logged by caller on command failure */
      });
      if (!client.isOpen) {
        await client.connect();
      }
      _redis = client as unknown as RedisLike;
      return _redis;
    } catch {
      invalidateRedis();
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
    const count = await atomicIncrementWithTtl(client, rkey, 60_000);
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
    invalidateRedis();
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
  const redisUrl = (env.XCELSIOR_MCP_REDIS_URL || env.MCP_REDIS_URL || env.REDIS_URL || "").trim();
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
    launchPerHour: Number(env.XCELSIOR_MCP_LAUNCH_ATTEMPTS_PER_HOUR || "20"),
    serverlessPerMinute: Number(env.XCELSIOR_MCP_SERVERLESS_INVOCATIONS_PER_MIN || "120"),
    maxWatchesPerPrincipal: Number(env.XCELSIOR_MCP_MAX_WATCHES_PER_PRINCIPAL || "3"),
  };
}
