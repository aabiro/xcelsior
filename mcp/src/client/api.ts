import { ApiError } from "./errors.js";

export interface ApiClientOptions {
  baseUrl: string;
  bearer: string;
}

export class XcelsiorApiClient {
  constructor(private readonly opts: ApiClientOptions) {}

  get bearer(): string {
    return this.opts.bearer;
  }

  get baseUrl(): string {
    return this.opts.baseUrl;
  }

  async get<T = unknown>(path: string, query?: Record<string, string | number | undefined>): Promise<T> {
    const url = new URL(path.startsWith("http") ? path : `${this.opts.baseUrl}${path}`);
    if (query) {
      for (const [k, v] of Object.entries(query)) {
        if (v !== undefined && v !== "") url.searchParams.set(k, String(v));
      }
    }
    return this.request<T>(url.toString(), { method: "GET" });
  }

  async post<T = unknown>(path: string, body?: unknown): Promise<T> {
    const url = path.startsWith("http") ? path : `${this.opts.baseUrl}${path}`;
    return this.request<T>(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  }

  private async request<T>(url: string, init: RequestInit): Promise<T> {
    const res = await fetch(url, {
      ...init,
      headers: {
        Authorization: `Bearer ${this.opts.bearer}`,
        Accept: "application/json",
        ...(init.headers as Record<string, string> | undefined),
      },
    });
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
      throw new ApiError(`Request failed: ${res.status} ${res.statusText}`, res.status, parsed);
    }
    return parsed as T;
  }
}