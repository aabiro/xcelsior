import posthog from "posthog-js";

/** True when the client bundle was built with a PostHog project token. */
export const posthogEnabled = Boolean(
  process.env.NEXT_PUBLIC_POSTHOG_PROJECT_TOKEN?.trim(),
);

function ready(): boolean {
  return posthogEnabled && typeof window !== "undefined";
}

export function phIdentify(
  distinctId: string,
  properties?: Record<string, unknown>,
): void {
  if (!ready()) return;
  posthog.identify(distinctId, properties);
}

export function phReset(): void {
  if (!ready()) return;
  posthog.reset();
}

export function phCapture(event: string, properties?: Record<string, unknown>): void {
  if (!ready()) return;
  posthog.capture(event, properties);
}

export function phCaptureException(error: Error, properties?: Record<string, unknown>): void {
  if (!ready()) return;
  posthog.captureException(error, properties);
}