import { describe, expect, it, vi } from "vitest";
import {
  DEFAULT_NOTIFICATION_URL,
  markNotificationReadInBackground,
  sanitizeNotificationUrl,
  serializePushSubscription,
  syncPushSubscriptionWithServer,
} from "@/lib/pwa/web-push";

function jsonResponse(body: unknown, init: ResponseInit = {}) {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  });
}

function createSubscription(overrides: Partial<PushSubscriptionJSON> = {}) {
  const payload: PushSubscriptionJSON = {
    endpoint: "https://push.example.test/subscription/123",
    expirationTime: null,
    keys: {
      auth: "auth-key",
      p256dh: "p256dh-key",
    },
    ...overrides,
  };

  return {
    endpoint: payload.endpoint!,
    expirationTime: payload.expirationTime ?? null,
    toJSON: () => payload,
    unsubscribe: vi.fn().mockResolvedValue(true),
  } as unknown as PushSubscription;
}

function createRegistration(options: {
  existingSubscription?: PushSubscription | null;
  newSubscription?: PushSubscription;
} = {}) {
  const subscribe = vi.fn().mockResolvedValue(options.newSubscription ?? createSubscription());
  const getSubscription = vi.fn().mockResolvedValue(options.existingSubscription ?? null);

  return {
    pushManager: {
      getSubscription,
      subscribe,
    },
  } as unknown as ServiceWorkerRegistration;
}

describe("desktop PWA web push utilities", () => {
  it("sanitizes notification targets to same-origin routes", () => {
    expect(sanitizeNotificationUrl("/dashboard/instances/job-1", "https://app.xcelsior.ca")).toBe(
      "/dashboard/instances/job-1",
    );
    expect(
      sanitizeNotificationUrl("https://app.xcelsior.ca/dashboard/billing?topup=true", "https://app.xcelsior.ca"),
    ).toBe("/dashboard/billing?topup=true");
    expect(sanitizeNotificationUrl("https://evil.example/phish", "https://app.xcelsior.ca")).toBe(
      DEFAULT_NOTIFICATION_URL,
    );
    expect(sanitizeNotificationUrl("javascript:alert(1)", "https://app.xcelsior.ca")).toBe(
      DEFAULT_NOTIFICATION_URL,
    );
  });

  it("serializes a browser push subscription payload", () => {
    const subscription = createSubscription();

    expect(serializePushSubscription(subscription)).toEqual({
      endpoint: "https://push.example.test/subscription/123",
      expirationTime: null,
      keys: {
        auth: "auth-key",
        p256dh: "p256dh-key",
      },
    });
  });

  it("rejects malformed browser push subscriptions", () => {
    const subscription = createSubscription({
      keys: {
        auth: "",
        p256dh: "",
      },
    });

    expect(() => serializePushSubscription(subscription)).toThrow(
      "Push subscription is missing required encryption keys.",
    );
  });

  it("marks notifications as read with a keepalive request", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ ok: true }));

    await markNotificationReadInBackground("notif-123", fetchMock as typeof fetch);

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/notifications/notif-123/read",
      expect.objectContaining({
        method: "POST",
        keepalive: true,
        credentials: "include",
      }),
    );
  });

  it("returns early when no subscription exists and auto-create is disabled", async () => {
    const registration = createRegistration();
    const fetchMock = vi.fn();

    const subscription = await syncPushSubscriptionWithServer({
      registration,
      createIfMissing: false,
      fetchImpl: fetchMock as typeof fetch,
    });

    expect(subscription).toBeNull();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("creates and syncs a new push subscription when requested", async () => {
    const newSubscription = createSubscription();
    const registration = createRegistration({ newSubscription });
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (input === "/api/notifications/push/subscription" && init?.method === "POST") {
        return jsonResponse({ ok: true, subscription_id: "wps-123" });
      }

      throw new Error(`Unexpected request: ${String(input)} ${init?.method ?? "GET"}`);
    });

    const subscription = await syncPushSubscriptionWithServer({
      registration,
      vapidPublicKey: "AQAB",
      fetchImpl: fetchMock as typeof fetch,
    });

    expect(subscription).toBe(newSubscription);
    expect((registration.pushManager.getSubscription as ReturnType<typeof vi.fn>)).toHaveBeenCalledTimes(1);
    expect((registration.pushManager.subscribe as ReturnType<typeof vi.fn>)).toHaveBeenCalledWith(
      expect.objectContaining({
        userVisibleOnly: true,
        applicationServerKey: expect.any(Uint8Array),
      }),
    );
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/notifications/push/subscription",
      expect.objectContaining({
        method: "POST",
        credentials: "include",
      }),
    );
  });

  it("re-syncs an existing browser subscription without re-subscribing", async () => {
    const existingSubscription = createSubscription();
    const registration = createRegistration({ existingSubscription });
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (input === "/api/notifications/push/subscription" && !init?.method) {
        return jsonResponse({
          ok: true,
          configured: true,
          vapid_public_key: "AQAB",
          active_subscription_count: 0,
        });
      }

      if (input === "/api/notifications/push/subscription" && init?.method === "POST") {
        return jsonResponse({ ok: true, subscription_id: "wps-123" });
      }

      throw new Error(`Unexpected request: ${String(input)} ${init?.method ?? "GET"}`);
    });

    const subscription = await syncPushSubscriptionWithServer({
      registration,
      existingSubscription,
      createIfMissing: false,
      fetchImpl: fetchMock as typeof fetch,
    });

    expect(subscription).toBe(existingSubscription);
    expect((registration.pushManager.subscribe as ReturnType<typeof vi.fn>)).not.toHaveBeenCalled();
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "/api/notifications/push/subscription",
      expect.objectContaining({
        credentials: "include",
      }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "/api/notifications/push/subscription",
      expect.objectContaining({
        method: "POST",
        credentials: "include",
      }),
    );
  });
});
