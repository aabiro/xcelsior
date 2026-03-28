import { cn } from "@/lib/utils";

interface CanadaFlagProps {
  size?: "sm" | "md" | "lg";
  className?: string;
}

/** Inline SVG maple leaf flag badge. */
export function CanadaFlag({ size = "md", className }: CanadaFlagProps) {
  const dim = size === "sm" ? "h-4 w-4" : size === "lg" ? "h-8 w-8" : "h-6 w-6";

  return (
    <span className={cn("inline-flex items-center", className)} aria-label="Canadian">
      <svg
        viewBox="0 0 32 32"
        className={dim}
        role="img"
        aria-hidden="true"
      >
        {/* Red bars */}
        <rect x="0" y="0" width="8" height="32" fill="#dc2626" />
        <rect x="24" y="0" width="8" height="32" fill="#dc2626" />
        {/* White center */}
        <rect x="8" y="0" width="16" height="32" fill="#ffffff" />
        {/* Simplified maple leaf */}
        <path
          d="M16 6 L17.5 12 L22 10 L19.5 14 L24 16 L19 16 L20 20 L16 17 L12 20 L13 16 L8 16 L12.5 14 L10 10 L14.5 12 Z"
          fill="#dc2626"
        />
        {/* Stem */}
        <rect x="15" y="20" width="2" height="4" fill="#dc2626" />
      </svg>
    </span>
  );
}
