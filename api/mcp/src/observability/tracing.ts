import { context, propagation, trace, SpanStatusCode } from "@opentelemetry/api";

const tracer = trace.getTracer("xcelsior-mcp", "2.0.0");

export async function traced<T>(
  name: string,
  fn: (traceparent: string | undefined, traceId: string | undefined) => Promise<T>,
): Promise<T> {
  return tracer.startActiveSpan(name, async (span) => {
    try {
      const carrier: Record<string, string> = {};
      propagation.inject(context.active(), carrier);
      const spanTraceId = span.spanContext().traceId;
      return await fn(
        carrier.traceparent,
        spanTraceId && spanTraceId !== "0".repeat(32) ? spanTraceId : undefined,
      );
    } catch (error) {
      span.recordException(error as Error);
      span.setStatus({ code: SpanStatusCode.ERROR });
      throw error;
    } finally {
      span.end();
    }
  });
}
