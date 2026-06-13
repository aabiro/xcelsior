import { cn } from "@/lib/utils";

/**
 * Polished "AI spark" mark — a four-point star with a small accent star,
 * the convention used across modern AI surfaces. Replaces the generic lucide
 * Sparkles glyph on the assistant rail so it reads as an intentional brand
 * element rather than a default icon. Uses currentColor so it inherits the
 * button's text colour (and its active/hover states).
 */
export function AiSparkIcon({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      className={cn("shrink-0", className)}
      aria-hidden
    >
      {/* Primary four-point star */}
      <path
        d="M12 1.8c.38 3.07 1.05 4.96 2.18 6.1 1.13 1.13 3.02 1.8 6.02 2.1-3 .38-4.89 1.05-6.02 2.18-1.13 1.13-1.8 3.02-2.18 6.02-.38-3-1.05-4.89-2.18-6.02-1.13-1.13-3.02-1.8-6.02-2.18 3-.3 4.89-.97 6.02-2.1C10.95 6.76 11.62 4.87 12 1.8Z"
        fill="currentColor"
      />
      {/* Small accent star, top-right */}
      <path
        d="M19 2.6c.16 1.18.45 1.9.95 2.4.5.5 1.22.79 2.4.95-1.18.16-1.9.45-2.4.95-.5.5-.79 1.22-.95 2.4-.16-1.18-.45-1.9-.95-2.4-.5-.5-1.22-.79-2.4-.95 1.18-.16 1.9-.45 2.4-.95.5-.5.79-1.22.95-2.4Z"
        fill="currentColor"
        opacity="0.65"
      />
    </svg>
  );
}
