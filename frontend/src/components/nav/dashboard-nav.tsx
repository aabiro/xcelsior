"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  BarChart3,
  Calendar,
  ChevronDown,
  Cpu,
  CreditCard,
  DollarSign,
  FileCheck,
  HardDrive,
  Layers,
  LayoutDashboard,
  Monitor,
  Package,
  Server,
  Shield,
  Sparkles,
  Store,
  TrendingUp,
  Users,
  Zap,
  type LucideIcon,
} from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";
import { cn } from "@/lib/utils";

const GROUPS_STORAGE_KEY = "xcelsior-nav-groups-open";

type NavItemDef = {
  href: string;
  key: string;
  icon: LucideIcon;
  badge?: string;
  roles?: string[];
  serverlessOnly?: boolean;
};

type NavGroupDef = {
  id: string;
  labelKey: string;
  items: NavItemDef[];
  roles?: string[];
};

/** Pinned — always visible, never tucked in a group. */
const PINNED_ITEMS: NavItemDef[] = [
  { href: "/dashboard", key: "dash.overview", icon: LayoutDashboard },
  { href: "/dashboard/ai", key: "dash.ai", icon: Sparkles, badge: "New" },
];

/**
 * Grouped nav — order follows the user journey:
 * Compute → Storage → Business → Insights → Trust → Admin
 */
const NAV_GROUPS: NavGroupDef[] = [
  {
    id: "compute",
    labelKey: "dash.nav.compute",
    items: [
      { href: "/dashboard/hosts", key: "dash.hosts", icon: Server },
      { href: "/dashboard/instances", key: "dash.instances", icon: Monitor },
      { href: "/dashboard/templates", key: "dash.templates", icon: Layers },
      { href: "/dashboard/inference", key: "dash.inference", icon: Zap, serverlessOnly: true },
      { href: "/dashboard/hpc", key: "dash.hpc", icon: Cpu, roles: ["admin"] },
    ],
  },
  {
    id: "storage",
    labelKey: "dash.nav.storage",
    items: [
      { href: "/dashboard/volumes", key: "dash.volumes", icon: HardDrive },
      { href: "/dashboard/artifacts", key: "dash.artifacts", icon: Package },
    ],
  },
  {
    id: "business",
    labelKey: "dash.nav.business",
    items: [
      { href: "/dashboard/billing", key: "dash.billing", icon: CreditCard },
      { href: "/dashboard/marketplace", key: "dash.marketplace", icon: Store },
      { href: "/dashboard/spot-pricing", key: "dash.spot_pricing", icon: TrendingUp },
      { href: "/dashboard/earnings", key: "dash.earnings", icon: DollarSign },
    ],
  },
  {
    id: "insights",
    labelKey: "dash.nav.insights",
    items: [
      { href: "/dashboard/telemetry", key: "dash.telemetry", icon: Activity },
      { href: "/dashboard/analytics", key: "dash.analytics", icon: BarChart3 },
      { href: "/dashboard/events", key: "dash.events", icon: Calendar },
    ],
  },
  {
    id: "trust",
    labelKey: "dash.nav.trust",
    items: [
      // Reputation page hidden for now — Phase 1 uses a flat platform fee, so
      // reputation tiers don't yet affect economics. Re-add when incentives return.
      { href: "/dashboard/compliance", key: "dash.compliance", icon: FileCheck },
      { href: "/dashboard/trust", key: "dash.trust", icon: Shield },
    ],
  },
  {
    id: "platform",
    labelKey: "dash.nav.platform",
    roles: ["admin"],
    items: [{ href: "/dashboard/admin", key: "dash.admin", icon: Users, roles: ["admin"] }],
  },
];

function readGroupsOpen(): Record<string, boolean> {
  if (typeof window === "undefined") return {};
  try {
    const raw = localStorage.getItem(GROUPS_STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Record<string, boolean>) : {};
  } catch {
    return {};
  }
}

function isItemActive(pathname: string, href: string): boolean {
  if (href === "/dashboard") return pathname === "/dashboard";
  return pathname === href || pathname.startsWith(`${href}/`);
}

function filterItem(
  item: NavItemDef,
  canAccessRole: (r: string) => boolean,
  showServerless: boolean,
): boolean {
  if (item.serverlessOnly && !showServerless) return false;
  if (item.roles && !item.roles.some(canAccessRole)) return false;
  return true;
}

function NavLink({
  item,
  label,
  active,
  collapsed,
  mobile,
  onNavigate,
  variant = "default",
}: {
  item: NavItemDef;
  label: string;
  active: boolean;
  collapsed: boolean;
  mobile: boolean;
  onNavigate?: () => void;
  variant?: "default" | "ai";
}) {
  const showLabel = mobile || !collapsed;
  return (
    <Link
      href={item.href}
      onClick={onNavigate}
      title={!showLabel ? label : undefined}
      className={cn(
        "flex items-center gap-3 rounded-lg px-3 py-2 text-[15px] transition-colors",
        active
          ? variant === "ai"
            ? "bg-accent-violet/12 text-accent-violet nav-active"
            : "bg-accent-cyan/8 text-accent-cyan nav-active"
          : variant === "ai"
            ? "text-text-secondary hover:bg-accent-violet/8 hover:text-accent-violet"
            : "text-text-secondary hover:bg-surface-hover hover:text-text-primary",
      )}
    >
      <item.icon
        className={cn(
          "h-[18px] w-[18px] shrink-0",
          active && (variant === "ai" ? "drop-shadow-[0_0_4px_rgba(124,58,237,0.45)]" : "drop-shadow-[0_0_4px_rgba(0,212,255,0.5)]"),
        )}
      />
      {showLabel && (
        <div className="flex min-w-0 flex-1 items-center gap-2">
          <span className="truncate">{label}</span>
          {item.badge && (
            <span className="shrink-0 rounded-full bg-accent-violet/12 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-accent-violet/90">
              {item.badge}
            </span>
          )}
        </div>
      )}
    </Link>
  );
}

export function DashboardNav({
  collapsed,
  mobile,
  showServerless,
  canAccessRole,
  t,
  onNavigate,
}: {
  collapsed: boolean;
  mobile: boolean;
  showServerless: boolean;
  canAccessRole: (role: string) => boolean;
  t: (key: string) => string;
  onNavigate?: () => void;
}) {
  const pathname = usePathname();
  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>({});
  const [hydrated, setHydrated] = useState(false);

  const visibleGroups = useMemo(
    () =>
      NAV_GROUPS.map((g) => ({
        ...g,
        items: g.items.filter((item) => filterItem(item, canAccessRole, showServerless)),
      })).filter((g) => {
        if (g.roles && !g.roles.some(canAccessRole)) return false;
        return g.items.length > 0;
      }),
    [canAccessRole, showServerless],
  );

  const activeGroupId = useMemo(() => {
    for (const g of visibleGroups) {
      if (g.items.some((item) => isItemActive(pathname, item.href))) return g.id;
    }
    return null;
  }, [pathname, visibleGroups]);

  useEffect(() => {
    const stored = readGroupsOpen();
    const next: Record<string, boolean> = { ...stored };
    for (const g of visibleGroups) {
      if (activeGroupId === g.id) next[g.id] = true;
      if (next[g.id] === undefined) next[g.id] = g.id === "compute";
    }
    setOpenGroups(next);
    setHydrated(true);
  }, [activeGroupId, visibleGroups]);

  const toggleGroup = useCallback((id: string) => {
    setOpenGroups((prev) => {
      const next = { ...prev, [id]: !prev[id] };
      try {
        localStorage.setItem(GROUPS_STORAGE_KEY, JSON.stringify(next));
      } catch {
        /* noop */
      }
      return next;
    });
  }, []);

  const expanded = mobile || !collapsed;

  return (
    <nav className="flex-1 overflow-y-auto overflow-x-hidden py-3 px-2 space-y-1">
      {/* Pinned */}
      <div className="space-y-0.5 pb-2">
        {PINNED_ITEMS.map((item) => (
          <NavLink
            key={item.href}
            item={item}
            label={t(item.key)}
            active={isItemActive(pathname, item.href)}
            collapsed={collapsed}
            mobile={mobile}
            onNavigate={onNavigate}
            variant={item.href === "/dashboard/ai" ? "ai" : "default"}
          />
        ))}
      </div>

      {expanded ? (
        visibleGroups.map((group) => {
          const isOpen = hydrated ? !!openGroups[group.id] : group.id === "compute";
          const groupActive = group.items.some((item) => isItemActive(pathname, item.href));
          return (
            <div key={group.id} className="pt-1">
              <button
                type="button"
                onClick={() => toggleGroup(group.id)}
                className={cn(
                  "flex w-full items-center gap-2 rounded-lg px-3 py-1.5 text-[11px] font-semibold uppercase tracking-widest transition-colors",
                  groupActive ? "text-accent-cyan/80" : "text-text-muted hover:text-text-secondary",
                )}
              >
                <span className="flex-1 text-left">{t(group.labelKey)}</span>
                <ChevronDown
                  className={cn("h-3.5 w-3.5 shrink-0 transition-transform duration-200", isOpen && "rotate-180")}
                />
              </button>
              <AnimatePresence initial={false}>
                {isOpen && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.18, ease: "easeOut" }}
                    className="overflow-hidden"
                  >
                    <div className="space-y-0.5 pt-0.5 pb-1 pl-1">
                      {group.items.map((item) => (
                        <NavLink
                          key={item.href}
                          item={item}
                          label={t(item.key)}
                          active={isItemActive(pathname, item.href)}
                          collapsed={false}
                          mobile={mobile}
                          onNavigate={onNavigate}
                        />
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          );
        })
      ) : (
        // Collapsed: show every nav icon (flattened groups) — no "More" popout.
        <div className="space-y-0.5">
          {visibleGroups
            .flatMap((group) => group.items)
            .map((item) => (
              <NavLink
                key={item.href}
                item={item}
                label={t(item.key)}
                active={isItemActive(pathname, item.href)}
                collapsed
                mobile={false}
                onNavigate={onNavigate}
              />
            ))}
        </div>
      )}
    </nav>
  );
}