/// <reference no-default-lib="true" />
/// <reference lib="esnext" />
/// <reference lib="webworker" />

import type { PrecacheEntry, SerwistGlobalConfig } from "serwist";
import { Serwist } from "serwist";
import { desktopRuntimeCaching } from "@/lib/pwa/runtime-caching";
import {
  DEFAULT_NOTIFICATION_URL,
  WEB_PUSH_LIFECYCLE_MESSAGE_TYPE,
  markNotificationReadInBackground,
  sanitizeNotificationUrl,
  syncPushSubscriptionWithServer,
} from "@/lib/pwa/web-push";

declare global {
  interface WorkerGlobalScope extends SerwistGlobalConfig {
    __SW_MANIFEST: (PrecacheEntry | string)[] | undefined;
  }
}

declare const self: ServiceWorkerGlobalScope;

type PushPayload = {
  title?: string;
  body?: string;
  icon?: string;
  badge?: string;
  tag?: string;
  url?: string;
  data?: Record<string, unknown>;
};

const serwist = new Serwist({
  precacheEntries: self.__SW_MANIFEST,
  skipWaiting: true,
  clientsClaim: true,
  navigationPreload: true,
  runtimeCaching: desktopRuntimeCaching,
  fallbacks: {
    entries: [
      {
        url: "/~offline",
        matcher({ request }) {
          return request.destination === "document";
        },
      },
    ],
  },
});

function parsePushPayload(data: PushMessageData | null): PushPayload {
  if (!data) return {};

  try {
    return data.json() as PushPayload;
  } catch {
    const body = data.text();
    return body ? { body } : {};
  }
}

async function broadcastWebPushLifecycleEvent(eventType: "received" | "clicked", payload: {
  notificationId: string | null;
  url: string;
  data?: Record<string, unknown>;
}) {
  const clients = await self.clients.matchAll({
    type: "window",
    includeUncontrolled: true,
  });

  for (const client of clients) {
    client.postMessage({
      type: WEB_PUSH_LIFECYCLE_MESSAGE_TYPE,
      event: eventType,
      timestamp: Date.now(),
      notificationId: payload.notificationId,
      url: payload.url,
      data: payload.data,
    });
  }
}

async function focusOrOpenWindow(url: string) {
  const targetPath = sanitizeNotificationUrl(url, self.location.origin);
  const targetUrl = new URL(targetPath, self.location.origin).href;
  const clients = await self.clients.matchAll({
    type: "window",
    includeUncontrolled: true,
  });

  const sameOriginClients = clients.filter((client) => new URL(client.url).origin === self.location.origin);
  const exactClient = sameOriginClients.find((client) => client.url === targetUrl);
  if (exactClient) {
    await exactClient.focus();
    return;
  }

  await self.clients.openWindow(targetUrl);
}

self.addEventListener("push", (event) => {
  const payload = parsePushPayload(event.data);
  const url = sanitizeNotificationUrl(payload.url ?? DEFAULT_NOTIFICATION_URL, self.location.origin);
  const notificationData: Record<string, unknown> & { url: string } = {
    ...(payload.data ?? {}),
    url,
  };

  event.waitUntil(
    self.registration.showNotification(payload.title ?? "Xcelsior", {
      body: payload.body ?? "You have a new Xcelsior notification.",
      icon: payload.icon ?? "/xcelsior_icon_192x192.png",
      badge: payload.badge ?? "/xcelsior_icon_192x192.png",
      tag: payload.tag ?? "xcelsior-notification",
      data: notificationData,
    }).then(() =>
      broadcastWebPushLifecycleEvent("received", {
        notificationId:
          typeof notificationData.notification_id === "string" ? notificationData.notification_id : null,
        url,
        data: payload.data,
      }),
    ),
  );
});

self.addEventListener("notificationclick", (event) => {
  event.notification.close();

  const url =
    typeof event.notification.data?.url === "string" ? event.notification.data.url : DEFAULT_NOTIFICATION_URL;
  const notificationId = event.notification.data?.notification_id;

  event.waitUntil(
    Promise.allSettled([
      focusOrOpenWindow(url),
      markNotificationReadInBackground(notificationId),
      broadcastWebPushLifecycleEvent("clicked", {
        notificationId: typeof notificationId === "string" ? notificationId : null,
        url,
        data:
          event.notification.data && typeof event.notification.data === "object"
            ? { ...(event.notification.data as Record<string, unknown>) }
            : undefined,
      }),
    ]).then(() => undefined),
  );
});

self.addEventListener("pushsubscriptionchange", (event) => {
  event.waitUntil(
    syncPushSubscriptionWithServer({
      registration: self.registration,
      existingSubscription: event.newSubscription,
      createIfMissing: true,
      fetchImpl: fetch,
    }).catch((error) => {
      console.warn("Failed to refresh web push subscription", error);
    }),
  );
});

serwist.addEventListeners();
