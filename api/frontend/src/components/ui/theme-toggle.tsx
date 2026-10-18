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
        "dashboard-site-theme-toggle",
        className
      )}
      title={isDark ? "Switch to light mode" : "Switch to dark mode"}
      aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
      aria-pressed={!isDark}
    >
      <Sun className="dashboard-site-theme-sun" />
      <Moon className="dashboard-site-theme-moon" />
      <span className="dashboard-site-theme-knob" aria-hidden />
    </button>
  );
}
