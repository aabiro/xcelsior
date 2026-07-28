import { NodeSDK } from "@opentelemetry/sdk-node";
import { HttpInstrumentation } from "@opentelemetry/instrumentation-http";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";

const endpoint = process.env.OTEL_EXPORTER_OTLP_ENDPOINT?.replace(/\/$/, "");
const sdk = new NodeSDK({
  serviceName: process.env.OTEL_SERVICE_NAME || "xcelsior-mcp",
  traceExporter: endpoint ? new OTLPTraceExporter({ url: `${endpoint}/v1/traces` }) : undefined,
  instrumentations: [new HttpInstrumentation()],
});

sdk.start();
for (const signal of ["SIGTERM", "SIGINT"] as const) {
  process.once(signal, () => { void sdk.shutdown(); });
}
