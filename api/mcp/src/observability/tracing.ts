import { AsyncLocalStorage } from "node:async_hooks";
import { randomUUID } from "node:crypto";
import { context, propagation, trace, SpanStatusCode } from "@opentelemetry/api";

const tracer = trace.getTracer("xcelsior-mcp", "2.0.0");
const toolTrace = new AsyncLocalStorage<string>();

export function currentToolTraceparent(): string | undefined {
  return toolTrace.getStore();
}

export async function traced<T>(
  name: string,
  fn: (traceparent: string | undefined, traceId: string | undefined) => Promise<T>,
): Promise<T> {
  return tracer.startActiveSpan(name, async (span) => {
    try {
      const carrier: Record<string, string> = {};
      propagation.inject(context.active(), carrier);
      const spanContext = span.spanContext();
      const recordingContext = spanContext.traceId !== "0".repeat(32);
      // Audit correlation also works when no telemetry exporter/provider is
      // configured. The request and its audit share this W3C context.
      const traceId = recordingContext ? spanContext.traceId : randomUUID().replaceAll("-", "");
      const spanId = recordingContext ? spanContext.spanId : randomUUID().replaceAll("-", "").slice(0, 16);
      const flags = recordingContext ? spanContext.traceFlags.toString(16).padStart(2, "0") : "00";
      const parent = carrier.traceparent || `00-${traceId}-${spanId}-${flags}`;
      return await toolTrace.run(parent, () => fn(parent, traceId));
    } catch (error) {
      span.recordException(error as Error);
      span.setStatus({ code: SpanStatusCode.ERROR });
      throw error;
    } finally {
      span.end();
    }
  });
}
