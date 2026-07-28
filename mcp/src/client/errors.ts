export interface ApiProblem {
  type?: string;
  title?: string;
  status?: number;
  detail?: string;
  code?: string;
  retryable?: boolean;
  retry_after_ms?: number;
  trace_id?: string;
  errors?: unknown[];
  [key: string]: unknown;
}

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly body?: unknown,
    public readonly problem?: ApiProblem,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export function formatApiError(err: unknown): string {
  if (err instanceof ApiError) {
    const detail = err.problem?.detail ?? JSON.stringify(err.body ?? {});
    return `Xcelsior API error (${err.status}): ${detail || err.message}`;
  }
  if (err instanceof Error) return err.message;
  return String(err);
}

export function apiProblem(err: unknown): Record<string, unknown> {
  if (err instanceof ApiError) {
    return {
      ok: false,
      error: "api_problem",
      status: err.status,
      code: err.problem?.code ?? "api_error",
      type: err.problem?.type,
      title: err.problem?.title,
      detail: err.problem?.detail ?? formatApiError(err),
      retryable: err.problem?.retryable ?? false,
      retry_after_ms: err.problem?.retry_after_ms,
      trace_id: err.problem?.trace_id,
      errors: err.problem?.errors ?? [],
    };
  }
  return {
    ok: false,
    error: "transport_error",
    code: "upstream_unavailable",
    detail: formatApiError(err),
    retryable: true,
  };
}
