import posthog from "posthog-js";

const posthogToken = process.env.NEXT_PUBLIC_POSTHOG_PROJECT_TOKEN?.trim();
if (posthogToken) {
  posthog.init(posthogToken, {
    api_host: "/ingest",
    ui_host: "https://us.posthog.com",
    defaults: "2026-05-30",
    capture_pageview: "history_change",
    capture_pageleave: true,
    capture_exceptions: true,
    // Nothing that identifies a visitor is stored until they accept cookies;
    // until then (and after Decline) events are counted cookieless.
    opt_out_capturing_by_default: true,
    cookieless_mode: "on_reject",
    debug: process.env.NODE_ENV === "development",
  });
}
