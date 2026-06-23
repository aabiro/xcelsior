"use client";

import { Moon, Sun } from "lucide-react";
import { useTheme } from "@/lib/theme";
import { useMounted } from "@/hooks/useMounted";
import { cn } from "@/lib/utils";

export function ThemeToggle({ className }: { className?: string }) {
  const { theme, toggleTheme } = useTheme();
  const mounted = useMounted();

  const isDark = !mounted || theme === "dark";

  return (
    <button
      onClick={toggleTheme}
      className={cn(
        "relative flex min-h-11 min-w-11 items-center justify-center rounded-lg text-text-muted hover:bg-surface-hover hover:text-text-primary transition-colors",
        className
      )}
      title={isDark ? "Switch to light mode" : "Switch to dark mode"}
      aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
    >
      {/* Status convention: the icon shows the mode you're currently in
          (dark → moon, light → sun). The tooltip/aria-label describes the action. */}
      {isDark ? (
        <Moon className="h-6 w-6" />
      ) : (
        <Sun className="h-6 w-6" />
      )}
    </button>
  );
}
