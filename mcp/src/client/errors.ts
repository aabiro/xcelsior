export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly body?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export function formatApiError(err: unknown): string {
  if (err instanceof ApiError) {
    const detail =
      typeof err.body === "object" && err.body !== null && "detail" in err.body
        ? String((err.body as { detail: unknown }).detail)
        : JSON.stringify(err.body ?? {});
    return `Xcelsior API error (${err.status}): ${detail || err.message}`;
  }
  if (err instanceof Error) return err.message;
  return String(err);
}