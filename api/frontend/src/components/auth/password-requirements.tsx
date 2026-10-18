"use client";

import { CheckCircle, Circle } from "lucide-react";
import { cn } from "@/lib/utils";

interface PasswordRequirementItem {
  key: string;
  label: string;
  satisfied: boolean;
}

export function PasswordRequirements({
  items,
  className,
}: {
  items: PasswordRequirementItem[];
  className?: string;
}) {
  return (
    <ul className={cn("space-y-2", className)}>
      {items.map((item) => (
        <li
          key={item.key}
          data-satisfied={item.satisfied ? "true" : "false"}
          className={cn(
            "flex items-start gap-2 text-xs transition-colors",
            item.satisfied ? "text-emerald" : "text-text-muted",
          )}
        >
          {item.satisfied ? (
            <CheckCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
          ) : (
            <Circle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
          )}
          <span>{item.label}</span>
        </li>
      ))}
    </ul>
  );
}
