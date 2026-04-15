import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/* ── Fun random instance name generator (adjective-noun) ─────────── */

const ADJECTIVES = [
  "cosmic", "stellar", "blazing", "turbo", "hyper", "mega", "ultra", "super",
  "atomic", "quantum", "cyber", "neon", "nitro", "flash", "swift", "rapid",
  "epic", "mighty", "bold", "fierce", "primal", "rogue", "astral", "nova",
  "phantom", "lucid", "frosty", "golden", "shadow", "thunder", "crimson",
  "radiant", "vivid", "ignited", "soaring",
];

const NOUNS = [
  "falcon", "phoenix", "dragon", "panther", "wolf", "hawk", "tiger", "viper",
  "cobra", "raptor", "mustang", "raven", "lynx", "storm", "blaze", "comet",
  "nebula", "forge", "pulse", "nexus", "spark", "orbit", "titan", "matrix",
  "apex", "prism", "flux", "surge", "vertex", "cipher", "sentinel", "spectre",
  "saber", "talon", "ember",
];

export function generateFunName(): string {
  const adj = ADJECTIVES[Math.floor(Math.random() * ADJECTIVES.length)];
  const noun = NOUNS[Math.floor(Math.random() * NOUNS.length)];
  return `${adj}-${noun}`;
}
