import type { CSSProperties } from "react";
import { cn } from "@/lib/utils";

/** The real marks, one path each, in public/logos/clients (from Simple Icons). */
const CLIENT_LOGOS = {
  claude: "/logos/clients/claude.svg",
  cursor: "/logos/clients/cursor.svg",
  vscode: "/logos/clients/vscode.svg",
  github: "/logos/clients/github.svg",
  copilot: "/logos/clients/github-copilot.svg",
} as const;

export type ClientLogoName = keyof typeof CLIENT_LOGOS;

/** Which mark a client id from lib/mcp.ts or an agent tab wears, if any. */
export function clientLogoFor(id: string): ClientLogoName | null {
  if (id === "claude" || id === "claude-code") return "claude";
  if (id === "copilot-cli") return "copilot";
  return id in CLIENT_LOGOS ? (id as ClientLogoName) : null;
}

// Each mark gets its own pair of theme accents rather than the vendor's
// palette. Marketing tokens first, dashboard tokens as the fallback.
const CYAN = "var(--cyan, var(--color-accent-cyan))";
const VIOLET = "var(--violet-2, var(--color-accent-violet))";
const GREEN = "var(--green, var(--color-emerald))";
const GOLD = "var(--gold, var(--color-accent-gold))";
const CORAL = "var(--coral, var(--color-accent-orange))";

const CLIENT_TONES: Record<ClientLogoName, [string, string]> = {
  claude: [GOLD, CORAL],
  cursor: [CYAN, GREEN],
  vscode: [CYAN, VIOLET],
  github: [VIOLET, CYAN],
  copilot: [GREEN, CYAN],
};

/**
 * A client's real logo used as a mask and filled with theme colours, so it
 * follows light and dark. A parent can override the fill with
 * `--client-logo-fill`.
 */
export function ClientLogo({
  name,
  size = 18,
  className,
}: {
  name: ClientLogoName;
  size?: number;
  className?: string;
}) {
  const mask = `url(${CLIENT_LOGOS[name]}) center / contain no-repeat`;
  const [from, to] = CLIENT_TONES[name];
  const style: CSSProperties = {
    width: size,
    height: size,
    background: `var(--client-logo-fill, linear-gradient(135deg, ${from}, ${to}))`,
    mask,
    WebkitMask: mask,
  };
  return <span aria-hidden className={cn("inline-block shrink-0", className)} style={style} />;
}
