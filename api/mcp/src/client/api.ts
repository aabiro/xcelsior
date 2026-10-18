import { ApiError, type ApiProblem } from "./errors.js";
import { context, propagation } from "@opentelemetry/api";
import createClient, { type Client } from "openapi-fetch";
import type { paths } from "./generated/api.js";
import { captureApiCall } from "./request-context.js";

export interface ApiClientOptions {
  baseUrl: string;
  bearer: string;
  defaultTimeoutMs?: number;
  maxReadRetries?: number;
}

export class XcelsiorApiClient {
  /** Generated v1 contract client for callers that use a literal OpenAPI path. */
  readonly v1: Client<paths>;

  constructor(private readonly opts: ApiClientOptions) {
    this.v1 = createClient<paths>({
      baseUrl: opts.baseUrl,
      headers: { Authorization: `Bearer ${opts.bearer}`, Accept: "application/json" },
    });
  }

  get bearer(): string {
    return this.opts.bearer;
  }

  get baseUrl(): string {
    return this.opts.baseUrl;
  }

  async get<T = unknown>(
    path: string,
    query?: Record<string, string | number | boolean | undefined>,
    options?: RequestOptions,
  ): Promise<T> {
    const url = new URL(path.startsWith("http") ? path : `${this.opts.baseUrl}${path}`);
    if (query) {
      for (const [k, v] of Object.entries(query)) {
        if (v !== undefined && v !== "") url.searchParams.set(k, String(v));
      }
    }
    return this.request<T>(url.toString(), { method: "GET" }, { ...options, retry: options?.retry ?? "safe" });
  }

  async post<T = unknown>(path: string, body?: unknown, options?: RequestOptions): Promise<T> {
    const url = path.startsWith("http") ? path : `${this.opts.baseUrl}${path}`;
    return this.request<T>(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    }, options);
  }

  async put<T = unknown>(path: string, body?: unknown, options?: RequestOptions): Promise<T> {
    return this.write<T>("PUT", path, body, options);
  }

  async patch<T = unknown>(path: string, body?: unknown, options?: RequestOptions): Promise<T> {
    return this.write<T>("PATCH", path, body, options);
  }

  async delete<T = unknown>(path: string, body?: unknown, options?: RequestOptions): Promise<T> {
    return this.write<T>("DELETE", path, body, options);
  }

  private async write<T>(method: string, path: string, body?: unknown, options?: RequestOptions): Promise<T> {
    const url = path.startsWith("http") ? path : `${this.opts.baseUrl}${path}`;
    return this.request<T>(url, {
      method,
      headers: { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    }, options);
  }

  private async request<T>(url: string, init: RequestInit, options: RequestOptions = {}): Promise<T> {
    const retries = options.retry === "safe" || (options.retry === "idempotent" && options.idempotencyKey)
      ? (options.maxRetries ?? this.opts.maxReadRetries ?? 2)
      : 0;
    let lastError: unknown;
    for (let attempt = 0; attempt <= retries; attempt += 1) {
      try {
        return await this.requestOnce<T>(url, init, options);
      } catch (error) {
        lastError = error;
        if (!isTransient(error) || attempt === retries) throw error;
        await new Promise((resolve) => setTimeout(resolve, Math.min(500, 40 * 2 ** attempt + Math.random() * 25)));
      }
    }
    throw lastError;
  }

  private async requestOnce<T>(url: string, init: RequestInit, options: RequestOptions): Promise<T> {
    const ctrl = new AbortController();
    const timeout = options.timeoutMs ?? this.opts.defaultTimeoutMs ?? 15_000;
    const timer = setTimeout(() => ctrl.abort(new Error(`API deadline exceeded after ${timeout}ms`)), timeout);
    const signal = options.signal ? AbortSignal.any([ctrl.signal, options.signal]) : ctrl.signal;
    let res: Response;
    const traceHeaders: Record<string, string> = {};
    propagation.inject(context.active(), traceHeaders);
    try {
      res = await fetch(url, {
        ...init,
        signal,
        headers: {
          Authorization: `Bearer ${this.opts.bearer}`,
          Accept: "application/json",
          ...(options.traceparent ? { traceparent: options.traceparent } : {}),
          ...traceHeaders,
          ...(options.idempotencyKey ? { "Idempotency-Key": options.idempotencyKey } : {}),
          ...(init.headers as Record<string, string> | undefined),
        },
      });
    } finally {
      clearTimeout(timer);
    }
    const text = await res.text();
    let parsed: unknown = null;
    if (text) {
      try {
        parsed = JSON.parse(text);
      } catch {
        parsed = text;
      }
    }
    if (!res.ok) {
      const problem = isProblem(parsed) ? parsed : undefined;
      captureApiCall({
        route: new URL(url).pathname,
        method: String(init.method || "GET"),
        status: res.status,
        problemType: problem?.type || problem?.code,
      });
      throw new ApiError(`Request failed: ${res.status} ${res.statusText}`, res.status, parsed, problem);
    }
    captureApiCall({
      route: new URL(url).pathname,
      method: String(init.method || "GET"),
      status: res.status,
    });
    return parsed as T;
  }
}

export interface RequestOptions {
  timeoutMs?: number;
  signal?: AbortSignal;
  traceparent?: string;
  idempotencyKey?: string;
  retry?: "none" | "safe" | "idempotent";
  maxRetries?: number;
}

function isProblem(value: unknown): value is ApiProblem {
  return typeof value === "object" && value !== null &&
    ("type" in value || "code" in value || "detail" in value);
}

function isTransient(error: unknown): boolean {
  if (error instanceof ApiError) return [408, 425, 429, 502, 503, 504].includes(error.status);
  return error instanceof TypeError || (error instanceof Error && error.name === "AbortError");
}
