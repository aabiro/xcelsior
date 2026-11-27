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

/** Reads an error body in either shape the API answers with: an RFC 9457
 *  problem (`type`, `code` or `detail` at the top level), or the API's own
 *  envelope, `{ ok: false, error: { code, message, details? } }`, which every
 *  HTTPException and request-validation error uses. Before this, the envelope
 *  was not recognised: the code was lost, the detail was the body as JSON, and
 *  a diagnostic tool said only "failed". The detail names the fields that
 *  failed, so an agent can fix its request. */
export function problemFromBody(body: unknown, status: number): ApiProblem | undefined {
  if (typeof body !== "object" || body === null || Array.isArray(body)) return undefined;
  const value = body as Record<string, unknown>;
  if ("type" in value || "code" in value || "detail" in value) {
    const problem: ApiProblem = { ...value };
    const errors = Array.isArray(value.errors) ? value.errors
      : Array.isArray(value.detail) ? value.detail : undefined;
    if (errors) problem.errors = errors;
    const fields = errors ? describeFieldErrors(errors) : "";
    const text = typeof value.detail === "string" ? value.detail : "";
    if (fields || typeof value.detail !== "string") {
      problem.detail = joinDetail(text, fields) || JSON.stringify(value.detail ?? value);
    }
    return problem;
  }
  const error = value.error;
  if (typeof error === "object" && error !== null && !Array.isArray(error)) {
    const envelope = error as Record<string, unknown>;
    const details = Array.isArray(envelope.details) ? envelope.details : [];
    const message = typeof envelope.message === "string" ? envelope.message : "";
    return {
      status,
      ...(typeof envelope.code === "string" ? { code: envelope.code } : {}),
      detail: joinDetail(message, describeFieldErrors(details)) || JSON.stringify(envelope),
      errors: details,
      ...(typeof envelope.retryable === "boolean" ? { retryable: envelope.retryable } : {}),
    };
  }
  if (typeof error === "string" && error) {
    return { status, detail: error };
  }
  return undefined;
}

/** "name: Field required; gpu_count: must be at least 1", from FastAPI's
 *  `{loc, msg}` entries or the launch validator's `{field, detail}`. */
function describeFieldErrors(entries: unknown[]): string {
  const parts: string[] = [];
  for (const item of entries) {
    if (typeof item !== "object" || item === null) continue;
    const entry = item as Record<string, unknown>;
    const where = Array.isArray(entry.loc)
      ? entry.loc.filter((part) => part !== "body").join(".")
      : typeof entry.field === "string" ? entry.field : "";
    const what = [entry.msg, entry.detail, entry.message].find(
      (text): text is string => typeof text === "string" && text.length > 0,
    );
    if (what) parts.push(where ? `${where}: ${what}` : what);
  }
  const shown = parts.slice(0, 5);
  if (parts.length > shown.length) shown.push(`and ${parts.length - shown.length} more`);
  return shown.join("; ");
}

function joinDetail(message: string, fields: string): string {
  if (message && fields) return `${message}: ${fields}`;
  return message || fields;
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
