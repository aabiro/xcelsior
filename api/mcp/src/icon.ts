import { readFileSync } from "node:fs";
import type { Icon } from "@modelcontextprotocol/sdk/types.js";

/**
 * The server icon a client shows next to "xcelsior", read from `serverInfo.icons`
 * in the initialize response (VS Code renders it in its tool picker).
 *
 * It is the transparent gradient X from the brand system, so it reads on light
 * and dark editor themes alike. The PNG lives in `assets/` beside `dist/`
 * rather than inlined: the Docker image and the npx CLI ship the same bytes,
 * and the initialize response stays a URL instead of an 18 KB data URI.
 *
 * Clients only trust an icon from a source the server could have produced
 * itself — VS Code accepts an HTTP server's icon only from that server's own
 * authority, and a stdio server's only from `file:` (or `data:`). So the
 * hosted server serves the file at ICON_PATH and the CLI points at it on disk.
 */
export const ICON_PATH = "/icon.png";

const ICON_FILE = new URL("../assets/icon-gradient-512.png", import.meta.url);

let cached: Buffer | null | undefined;

function icon(src: string): Icon {
  return { src, mimeType: "image/png", sizes: ["512x512"] };
}

/** The icon bytes, or null when this build was shipped without `assets/`. */
export function readIcon(): Buffer | null {
  if (cached === undefined) {
    try {
      cached = readFileSync(ICON_FILE);
    } catch {
      cached = null;
    }
  }
  return cached;
}

/** Icons for the hosted server: served from the connector URL's own origin. */
export function httpIcons(resourceAudience: string): Icon[] {
  if (!readIcon()) return [];
  try {
    return [icon(new URL(ICON_PATH, resourceAudience).href)];
  } catch {
    return [];
  }
}

/** Icons for the stdio CLI: the file on disk. */
export function stdioIcons(): Icon[] {
  if (!readIcon()) return [];
  return [icon(ICON_FILE.href)];
}
