export function jsonText(data: unknown): {
  content: Array<{ type: "text"; text: string }>;
  structuredContent: Record<string, unknown>;
  isError?: boolean;
} {
  const structuredContent =
    typeof data === "object" && data !== null && !Array.isArray(data)
      ? data as Record<string, unknown>
      : { data };
  const isError = Boolean(structuredContent.error) || structuredContent.ok === false;
  return {
    content: [{ type: "text", text: JSON.stringify(data, null, 2) }],
    structuredContent,
    ...(isError ? { isError: true } : {}),
  };
}

export function structuredResult<T extends Record<string, unknown>>(
  data: T,
  summary?: string,
): {
  content: Array<{ type: "text"; text: string }>;
  structuredContent: T;
  isError?: boolean;
} {
  const isError = Boolean(data.error) || data.ok === false;
  return {
    content: [{ type: "text", text: summary ?? JSON.stringify(data, null, 2) }],
    structuredContent: data,
    ...(isError ? { isError: true } : {}),
  };
}
