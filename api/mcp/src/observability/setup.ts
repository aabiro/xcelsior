import { NodeSDK } from "@opentelemetry/sdk-node";
import { HttpInstrumentation } from "@opentelemetry/instrumentation-http";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";

const endpoint = process.env.OTEL_EXPORTER_OTLP_ENDPOINT?.replace(/\/$/, "");
// Passing `traceExporter: undefined` lets NodeSDK install its default exporter,
// which both sends to an operator-unconfigured endpoint and can hold SIGTERM
// open while it retries. No endpoint means tracing is deliberately disabled.
const sdk = endpoint
  ? new NodeSDK({
      serviceName: process.env.OTEL_SERVICE_NAME || "xcelsior-mcp",
      traceExporter: new OTLPTraceExporter({ url: `${endpoint}/v1/traces` }),
      instrumentations: [new HttpInstrumentation()],
    })
  : undefined;

sdk?.start();

let shutdownPromise: Promise<void> | undefined;

export function shutdownObservability(): Promise<void> {
  shutdownPromise ??= sdk?.shutdown() ?? Promise.resolve();
  return shutdownPromise;
}
