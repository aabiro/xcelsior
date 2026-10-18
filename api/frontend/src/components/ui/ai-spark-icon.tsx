import { cn } from "@/lib/utils";

/**
 * Polished "AI spark" mark for the assistant rail. Symmetric viewBox for
 * optical centering in the toggle bubble. Uses currentColor.
 */
export function AiSparkIcon({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      className={cn("block shrink-0", className)}
      aria-hidden
    >
      <path
        d="M12 2.5c.55 3.6 1.55 5.75 3.15 7.35 1.6 1.6 3.75 2.6 7.35 3.15-3.6.55-5.75 1.55-7.35 3.15-1.6 1.6-2.6 3.75-3.15 7.35-.55-3.6-1.55-5.75-3.15-7.35-1.6-1.6-3.75-2.6-7.35-3.15 3.6-.55 5.75-1.55 7.35-3.15 1.6-1.6 2.6-3.75 3.15-7.35Z"
        fill="currentColor"
      />
      <path
        d="M19.25 5.75c.2 1.25.5 2 .95 2.45.45.45 1.2.75 2.45.95-1.25.2-2 .5-2.45.95-.45.45-.75 1.2-.95 2.45-.2-1.25-.5-2-.95-2.45-.45-.45-1.2-.75-2.45-.95 1.25-.2 2-.5 2.45-.95.45-.44.75-1.2.95-2.45Z"
        fill="currentColor"
        opacity="0.65"
      />
    </svg>
  );
}