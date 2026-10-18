"use client";

import { useEffect, useState } from "react";
import { fetchOAuthProviders } from "@/lib/api";

export const OAUTH_PROVIDERS = ["github", "google", "huggingface", "facebook"] as const;
export type OAuthProvider = (typeof OAUTH_PROVIDERS)[number];

/**
 * The sign-in providers the server has configured, in display order.
 *
 * Every known provider used to be shown, so one with no app behind it
 * (Facebook) answered each click with "OAuth provider is not configured".
 * Until the list loads, and if it cannot be loaded, all are shown, so the
 * sign-in options never disappear because of a network hiccup.
 */
export function useOAuthProviders(): readonly OAuthProvider[] {
  const [configured, setConfigured] = useState<string[] | null>(null);
  useEffect(() => {
    let cancelled = false;
    fetchOAuthProviders()
      .then((body) => {
        if (!cancelled && Array.isArray(body.providers)) setConfigured(body.providers);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);
  return configured ? OAUTH_PROVIDERS.filter((p) => configured.includes(p)) : OAUTH_PROVIDERS;
}
