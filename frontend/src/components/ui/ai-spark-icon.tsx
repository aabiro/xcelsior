import { cn } from "@/lib/utils";

/**
 * Polished "AI spark" mark for the assistant rail. Tight viewBox trims
 * transparent padding so the glyph centers in the bubble. Uses currentColor.
 */
export function AiSparkIcon({ className }: { className?: string }) {
  return (
    <svg
      viewBox="8 0 14 22"
      fill="none"
      className={cn("block shrink-0", className)}
      aria-hidden
    >
      <path
        d="M10.8 2.2c.36 2.88.98 4.63 2.02 5.67 1.05 1.05 2.8 1.67 5.63 2.03-2.83.36-4.58.98-5.63 2.03-1.04 1.04-1.66 2.8-2.02 5.63-.36-2.83-.98-4.58-2.03-5.63-1.04-1.05-2.79-1.67-5.62-2.03 2.83-.36 4.58-.98 5.62-2.03 1.05-1.04 1.67-2.79 2.03-5.67Z"
        fill="currentColor"
      />
      <path
        d="M18.55 4.2c.14 1 .39 1.62.81 2.03.42.42 1.03.67 2.04.81-1.01.14-1.62.39-2.04.81-.42.42-.67 1.03-.81 2.04-.14-1.01-.39-1.62-.81-2.04-.42-.42-1.03-.67-2.03-.81 1-.14 1.61-.39 2.03-.81.42-.41.67-1.03.81-2.03Z"
        fill="currentColor"
        opacity="0.65"
      />
    </svg>
  );
}