"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ChevronRight } from "lucide-react";
import { useBreadcrumbLabels } from "@/lib/breadcrumb-labels";

const ROUTE_LABELS: Record<string, string> = {
  dashboard: "Dashboard",
  ai: "Xcel AI",
  hosts: "Hosts",
  instances: "Instances",
  templates: "Templates",
  volumes: "Volumes",
  telemetry: "Telemetry",
  billing: "Billing",
  marketplace: "Marketplace",
  "spot-pricing": "Spot Pricing",
  earnings: "Earnings",
  reputation: "Reputation",
  compliance: "Compliance",
  trust: "Trust",
  analytics: "Analytics",
  artifacts: "Artifacts",
  hpc: "HPC / Slurm",
  events: "Events",
  settings: "Settings",
  admin: "Admin",
  notifications: "Notifications",
  new: "New",
  inference: "Serverless",
  // Admin sub-tabs (most are handled by the title-case fallback; these need
  // explicit casing for acronyms / multi-word labels).
  "ai-insights": "AI Insights",
  "unit-economics": "Unit Economics",
  mcp: "Connect MCP",
};

/**
 * Title-case an unknown path segment ("unit-economics" → "Unit Economics").
 * Ids ("ae1d03c3") keep their case: capitalising one makes it look like a name.
 */
function prettifySegment(seg: string): string {
  const decoded = decodeURIComponent(seg);
  if (/\d/.test(decoded)) return decoded;
  return decoded
    .split("-")
    .map((w) => (w ? w.charAt(0).toUpperCase() + w.slice(1) : w))
    .join(" ");
}

export function Breadcrumb() {
  const pathname = usePathname();
  const pageLabels = useBreadcrumbLabels();
  const segments = pathname.split("/").filter(Boolean);

  if (segments.length <= 1) return null;

  const crumbs = segments.map((seg, i) => {
    const href = "/" + segments.slice(0, i + 1).join("/");
    const label = pageLabels.get(href) || ROUTE_LABELS[seg] || prettifySegment(seg);
    const isLast = i === segments.length - 1;
    return { href, label, isLast };
  });

  return (
    <nav aria-label="Breadcrumb" className="flex items-center gap-1 text-base">
      {crumbs.map((crumb, i) => (
        <span key={crumb.href} className="flex items-center gap-1">
          {i > 0 && <ChevronRight className="h-4 w-4 text-text-muted" />}
          {crumb.isLast ? (
            <span className="text-text-primary font-medium">{crumb.label}</span>
          ) : (
            <Link
              href={crumb.href}
              className="text-text-muted hover:text-text-primary transition-colors"
            >
              {crumb.label}
            </Link>
          )}
        </span>
      ))}
    </nav>
  );
}
