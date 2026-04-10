export interface WebPushSubscriptionStatus {
  ok: boolean;
  configured: boolean;
  vapid_public_key: string;
  active_subscription_count: number;
}

export interface WebPushSubscriptionPayload {
  endpoint: string;
  expirationTime: number | null;
  keys: {
    p256dh: string;
    auth: string;
  };
}

export interface SyncPushSubscriptionOptions {
  registration: ServiceWorkerRegistration;
  existingSubscription?: PushSubscription | null;
  createIfMissing?: boolean;
  vapidPublicKey?: string;
  fetchImpl?: typeof fetch;
}

export const WEB_PUSH_LIFECYCLE_MESSAGE_TYPE = "xcelsior:web-push-lifecycle";

export interface WebPushLifecycleMessage {
  type: typeof WEB_PUSH_LIFECYCLE_MESSAGE_TYPE;
  event: "received" | "clicked";
  timestamp: number;
  notificationId: string | null;
  url: string;
  data?: Record<string, unknown>;
}

const JSON_HEADERS = {
  "Content-Type": "application/json",
} as const;

export const DEFAULT_NOTIFICATION_URL = "/dashboard/notifications";

export function isWebPushLifecycleMessage(value: unknown): value is WebPushLifecycleMessage {
  if (!value || typeof value !== "object") return false;

  const candidate = value as Record<string, unknown>;
  return (
    candidate.type === WEB_PUSH_LIFECYCLE_MESSAGE_TYPE &&
    (candidate.event === "received" || candidate.event === "clicked") &&
    typeof candidate.timestamp === "number" &&
    typeof candidate.url === "string"
  );
}

async function requestJson<T>(
  input: string,
  init: RequestInit = {},
  fetchImpl: typeof fetch = fetch,
): Promise<T> {
  const headers = new Headers(init.headers ?? undefined);
  if (!headers.has("Content-Type") && init.method && init.method !== "GET" && init.method !== "HEAD") {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetchImpl(input, {
    credentials: "include",
    ...init,
    headers,
  });

  if (!response.ok) {
    throw new Error(`Request failed: ${response.status} ${response.statusText}`);
  }

  return response.json() as Promise<T>;
}

export function sanitizeNotificationUrl(rawUrl: unknown, origin: string): string {
  if (typeof rawUrl !== "string") return DEFAULT_NOTIFICATION_URL;

  const trimmed = rawUrl.trim();
  if (!trimmed) return DEFAULT_NOTIFICATION_URL;

  try {
    const resolved = new URL(trimmed, origin);
    if (resolved.origin !== origin) return DEFAULT_NOTIFICATION_URL;
    if (!/^https?:$/.test(resolved.protocol)) return DEFAULT_NOTIFICATION_URL;

    const path = `${resolved.pathname}${resolved.search}${resolved.hash}`;
    return path.startsWith("/") ? path : DEFAULT_NOTIFICATION_URL;
  } catch {
    return DEFAULT_NOTIFICATION_URL;
  }
}

export function urlBase64ToUint8Array(value: string) {
  const padding = "=".repeat((4 - (value.length % 4)) % 4);
  const base64 = `${value}${padding}`.replace(/-/g, "+").replace(/_/g, "/");
  const raw = atob(base64);
  return Uint8Array.from(raw, (char) => char.charCodeAt(0));
}

export function serializePushSubscription(subscription: Pick<PushSubscription, "endpoint" | "expirationTime" | "toJSON">): WebPushSubscriptionPayload {
  const payload = subscription.toJSON();
  if (!payload.endpoint || !payload.keys?.p256dh || !payload.keys?.auth) {
    throw new Error("Push subscription is missing required encryption keys.");
  }

  return {
    endpoint: payload.endpoint,
    expirationTime: payload.expirationTime ?? null,
    keys: {
      p256dh: payload.keys.p256dh,
      auth: payload.keys.auth,
    },
  };
}

export async function fetchPushSubscriptionStatus(
  fetchImpl: typeof fetch = fetch,
): Promise<WebPushSubscriptionStatus> {
  return requestJson<WebPushSubscriptionStatus>("/api/notifications/push/subscription", undefined, fetchImpl);
}

export async function upsertPushSubscription(
  payload: WebPushSubscriptionPayload,
  fetchImpl: typeof fetch = fetch,
): Promise<void> {
  await requestJson<{ ok: boolean; subscription_id: string }>(
    "/api/notifications/push/subscription",
    {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify(payload),
    },
    fetchImpl,
  );
}

export async function revokePushSubscriptionOnServer(
  endpoint: string,
  fetchImpl: typeof fetch = fetch,
): Promise<void> {
  await requestJson<{ ok: boolean; revoked: boolean }>(
    "/api/notifications/push/subscription",
    {
      method: "DELETE",
      headers: JSON_HEADERS,
      body: JSON.stringify({ endpoint }),
    },
    fetchImpl,
  );
}

export async function markNotificationReadInBackground(
  notificationId: unknown,
  fetchImpl: typeof fetch = fetch,
): Promise<void> {
  if (typeof notificationId !== "string" || !notificationId.trim()) return;

  await requestJson<{ ok: boolean }>(
    `/api/notifications/${encodeURIComponent(notificationId)}/read`,
    {
      method: "POST",
      keepalive: true,
    },
    fetchImpl,
  );
}

export async function syncPushSubscriptionWithServer({
  registration,
  existingSubscription,
  createIfMissing = true,
  vapidPublicKey,
  fetchImpl = fetch,
}: SyncPushSubscriptionOptions): Promise<PushSubscription | null> {
  const currentSubscription = existingSubscription ?? await registration.pushManager.getSubscription();
  if (!currentSubscription && !createIfMissing) {
    return null;
  }

  let resolvedVapidPublicKey = vapidPublicKey;
  if (!resolvedVapidPublicKey) {
    const status = await fetchPushSubscriptionStatus(fetchImpl);
    if (!status.configured || !status.vapid_public_key) {
      return null;
    }
    resolvedVapidPublicKey = status.vapid_public_key;
  }

  const subscription = currentSubscription ?? await registration.pushManager.subscribe({
    userVisibleOnly: true,
    applicationServerKey: urlBase64ToUint8Array(resolvedVapidPublicKey),
  });

  await upsertPushSubscription(serializePushSubscription(subscription), fetchImpl);
  return subscription;
}
