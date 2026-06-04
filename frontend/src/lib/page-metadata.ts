import type { Metadata } from "next";

const BASE = "https://xcelsior.ca";

/** Metadata for auth / utility pages that should not compete in search indexes. */
export function privatePageMetadata(title: string, path: string, description?: string): Metadata {
  const url = `${BASE}${path}`;
  return {
    title,
    description: description ?? `${title} — Xcelsior`,
    alternates: { canonical: url },
    robots: { index: false, follow: false },
    openGraph: { title: `${title} | Xcelsior`, url },
  };
}