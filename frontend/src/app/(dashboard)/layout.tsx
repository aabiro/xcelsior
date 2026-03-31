"use client";

import { useState, useEffect, useRef } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard, Server, Monitor, Activity, CreditCard,
  Store, DollarSign, ShieldCheck, Star, FileCheck,
  BarChart3, Package, Calendar, Settings, Users, ChevronLeft,
  ChevronRight, LogOut, Shield, Cpu, Menu, X, Key, ChevronDown,
  Zap, HardDrive, TrendingUp,
} from "lucide-react";
import { useAuth } from "@/lib/auth";
import { useLocale } from "@/lib/locale";
import { cn } from "@/lib/utils";
import { AnimatePresence, motion } from "framer-motion";
import { Breadcrumb } from "@/components/ui/breadcrumb";
import { ThemeToggle } from "@/components/ui/theme-toggle";
import { LocaleToggle } from "@/components/ui/locale-toggle";
import { ChatWidget } from "@/components/ChatWidget";
import { NotificationBell } from "@/components/NotificationBell";

const navItems: { href: string; key: string; icon: typeof LayoutDashboard; roles?: string[] }[] = [
  { href: "/dashboard", key: "dash.overview", icon: LayoutDashboard },
  { href: "/dashboard/hosts", key: "dash.hosts", icon: Server },
  { href: "/dashboard/instances", key: "dash.instances", icon: Monitor },
  { href: "/dashboard/telemetry", key: "dash.telemetry", icon: Activity },
  { href: "/dashboard/billing", key: "dash.billing", icon: CreditCard },
  { href: "/dashboard/marketplace", key: "dash.marketplace", icon: Store },
  { href: "/dashboard/spot-pricing", key: "dash.spot_pricing", icon: TrendingUp },
  { href: "/dashboard/inference", key: "dash.inference", icon: Zap },
  { href: "/dashboard/volumes", key: "dash.volumes", icon: HardDrive },
  { href: "/dashboard/earnings", key: "dash.earnings", icon: DollarSign },
  { href: "/dashboard/reputation", key: "dash.reputation", icon: Star },
  { href: "/dashboard/compliance", key: "dash.compliance", icon: FileCheck },
  { href: "/dashboard/trust", key: "dash.trust", icon: Shield },
  { href: "/dashboard/analytics", key: "dash.analytics", icon: BarChart3 },
  { href: "/dashboard/artifacts", key: "dash.artifacts", icon: Package },
  { href: "/dashboard/hpc", key: "dash.hpc", icon: Cpu, roles: ["admin", "provider"] },
  { href: "/dashboard/events", key: "dash.events", icon: Calendar },
  { href: "/dashboard/admin", key: "dash.admin", icon: Users, roles: ["admin"] },
];

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);
  const profileRef = useRef<HTMLDivElement>(null);
  const pathname = usePathname();
  const { user, loading: authLoading, logout } = useAuth();
  const { t } = useLocale();

  // Redirect to login if not authenticated
  useEffect(() => {
    if (!authLoading && !user) {
      window.location.href = `/login?redirect=${encodeURIComponent(pathname)}`;
    }
  }, [authLoading, user, pathname]);

  // Close mobile drawer on route change
  useEffect(() => { setMobileOpen(false); setProfileOpen(false); }, [pathname]);

  // Close profile dropdown on click outside
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (profileRef.current && !profileRef.current.contains(e.target as Node)) {
        setProfileOpen(false);
      }
    }
    if (profileOpen) document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [profileOpen]);

  // Show loading state while checking auth
  if (authLoading || !user) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-accent-red border-t-transparent" />
      </div>
    );
  }

  const sidebarContent = (mobile: boolean) => (
    <>
      {/* Logo */}
      <div className="flex h-14 items-center border-b border-border/60 px-4 justify-between">
        <Link href="/dashboard" className="flex items-center gap-2">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg" style={{ background: 'linear-gradient(135deg, #060a13 0%, #0d1a2a 100%)' }}>
            <svg viewBox="0 0 256 256" className="h-5 w-5" aria-hidden>
              <defs>
                <linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">
                  <stop offset="0%" stopColor="#00d4ff" />
                  <stop offset="60%" stopColor="#7c3aed" />
                  <stop offset="100%" stopColor="#dc2626" />
                </linearGradient>
              </defs>
              <path d="M60 56 L120 128 L60 200 L90 200 L128 152 L166 200 L196 200 L136 128 L196 56 L166 56 L128 104 L90 56Z" fill="url(#lg)" />
              <path d="M118 36 L128 20 L138 36Z" fill="#00d4ff" opacity="0.9" />
            </svg>
          </div>
          {(mobile || !collapsed) && (
            <>
              <span className="font-bold tracking-tight">Xcelsior</span>
              <span className="rounded bg-accent-cyan/10 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-accent-cyan">
                Beta
              </span>
            </>
          )}
        </Link>
        {mobile && (
          <button onClick={() => setMobileOpen(false)} className="text-text-muted hover:text-text-primary" aria-label="Close menu">
            <X className="h-5 w-5" />
          </button>
        )}
      </div>

      {/* Nav */}
      <nav className="flex-1 overflow-y-auto py-3 px-2 space-y-0.5">
        {navItems
          .filter((item) => !item.roles || item.roles.includes(user?.role ?? ""))
          .map((item) => {
          const active = pathname === item.href || (item.href !== "/dashboard" && pathname.startsWith(item.href));
          const label = t(item.key);
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors",
                active
                  ? "bg-accent-cyan/8 text-accent-cyan nav-active"
                  : "text-text-secondary hover:bg-surface-hover hover:text-text-primary"
              )}
              title={!mobile && collapsed ? label : undefined}
            >
              <item.icon className={`h-4 w-4 shrink-0 ${active ? 'drop-shadow-[0_0_4px_rgba(0,212,255,0.5)]' : ''}`} />
              {(mobile || !collapsed) && <span>{label}</span>}
            </Link>
          );
        })}
      </nav>

      {/* Collapse Toggle (desktop only) */}
      {!mobile && (
        <div className="border-t border-border p-2 space-y-0.5">
          <Link
            href="/dashboard/settings"
            className={cn(
              "flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors",
              pathname.startsWith("/dashboard/settings")
                ? "bg-accent-cyan/8 text-accent-cyan nav-active"
                : "text-text-secondary hover:bg-surface-hover hover:text-text-primary"
            )}
            title={collapsed ? t("dash.settings") : undefined}
          >
            <Settings className="h-4 w-4 shrink-0" />
            {!collapsed && <span>{t("dash.settings")}</span>}
          </Link>
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="flex w-full items-center justify-center rounded-lg px-3 py-2 text-text-muted hover:bg-surface-hover hover:text-text-primary"
          >
            {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
          </button>
        </div>
      )}
      {/* Settings link (mobile drawer) */}
      {mobile && (
        <div className="border-t border-border p-2">
          <Link
            href="/dashboard/settings"
            className={cn(
              "flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors",
              pathname.startsWith("/dashboard/settings")
                ? "bg-accent-cyan/8 text-accent-cyan nav-active"
                : "text-text-secondary hover:bg-surface-hover hover:text-text-primary"
            )}
          >
            <Settings className="h-4 w-4 shrink-0" />
            <span>{t("dash.settings")}</span>
          </Link>
        </div>
      )}
    </>
  );

  return (
    <div className="flex h-screen overflow-hidden bg-navy">
      {/* Desktop Sidebar */}
      <aside
        className={cn(
          "hidden md:flex flex-col border-r border-border/60 glass transition-all duration-200",
          collapsed ? "w-16" : "w-60"
        )}
      >
        {sidebarContent(false)}
      </aside>

      {/* Mobile Drawer Overlay */}
      <AnimatePresence>
        {mobileOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="fixed inset-0 z-40 bg-black/60 md:hidden"
              onClick={() => setMobileOpen(false)}
            />
            <motion.aside
              initial={{ x: -280 }}
              animate={{ x: 0 }}
              exit={{ x: -280 }}
              transition={{ type: "spring", damping: 25, stiffness: 300 }}
              className="fixed inset-y-0 left-0 z-50 flex w-[280px] flex-col border-r border-border/60 glass md:hidden"
            >
              {sidebarContent(true)}
            </motion.aside>
          </>
        )}
      </AnimatePresence>

      {/* Main */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Topbar */}
        <header className="flex h-14 items-center justify-between border-b border-border/60 glass px-4 md:px-6">
          {/* Mobile menu button */}
          <button
            className="md:hidden flex items-center justify-center h-9 w-9 rounded-lg text-text-secondary hover:bg-surface-hover hover:text-text-primary"
            onClick={() => setMobileOpen(true)}
            aria-label="Open menu"
          >
            <Menu className="h-5 w-5" />
          </button>
          <div className="hidden md:block">
            <Breadcrumb />
          </div>
          <div className="flex items-center gap-4">
            <LocaleToggle />
            <ThemeToggle />
            <div className="h-5 w-px bg-border hidden sm:block" />
            <NotificationBell />
            <div className="h-6 w-px bg-border hidden sm:block" />
            <div className="relative" ref={profileRef}>
              <button
                onClick={() => setProfileOpen(!profileOpen)}
                className="flex items-center gap-2 rounded-lg px-2 py-1.5 hover:bg-surface-hover transition-colors"
              >
                <div className="flex h-8 w-8 items-center justify-center rounded-full bg-accent-cyan/15 text-sm font-medium text-accent-cyan ring-1 ring-accent-cyan/20">
                  {user?.name?.[0]?.toUpperCase() || user?.email?.[0]?.toUpperCase() || "?"}
                </div>
                {user && (
                  <div className="hidden sm:block text-left">
                    <p className="text-sm font-medium leading-none">{user.name || user.email}</p>
                    <p className="text-xs text-text-muted">{user.role || "user"}</p>
                  </div>
                )}
                <ChevronDown className={cn("h-3.5 w-3.5 text-text-muted transition-transform hidden sm:block", profileOpen && "rotate-180")} />
              </button>
              {/* Profile dropdown */}
              <AnimatePresence>
                {profileOpen && (
                  <motion.div
                    initial={{ opacity: 0, y: -4 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -4 }}
                    transition={{ duration: 0.15 }}
                    className="absolute right-0 top-full mt-1 w-56 rounded-xl border border-border/60 glass shadow-xl z-50 overflow-hidden"
                  >
                    <div className="px-3 py-2.5 border-b border-border">
                      <p className="text-sm font-medium truncate">{user?.name || user?.email}</p>
                      <p className="text-xs text-text-muted truncate">{user?.email}</p>
                    </div>
                    <div className="py-1">
                      <Link
                        href="/dashboard/settings"
                        className="flex items-center gap-2.5 px-3 py-2 text-sm text-text-secondary hover:bg-surface-hover hover:text-text-primary transition-colors"
                        onClick={() => setProfileOpen(false)}
                      >
                        <Settings className="h-4 w-4" />
                        {t("dash.settings")}
                      </Link>
                      <Link
                        href="/dashboard/billing"
                        className="flex items-center gap-2.5 px-3 py-2 text-sm text-text-secondary hover:bg-surface-hover hover:text-text-primary transition-colors"
                        onClick={() => setProfileOpen(false)}
                      >
                        <CreditCard className="h-4 w-4" />
                        {t("dash.billing")}
                      </Link>
                      <Link
                        href="/dashboard/settings#api-keys"
                        className="flex items-center gap-2.5 px-3 py-2 text-sm text-text-secondary hover:bg-surface-hover hover:text-text-primary transition-colors"
                        onClick={() => setProfileOpen(false)}
                      >
                        <Key className="h-4 w-4" />
                        {t("dash.settings.api_keys") || "API Keys"}
                      </Link>
                    </div>
                    <div className="border-t border-border py-1">
                      <button
                        onClick={() => { setProfileOpen(false); void logout(); }}
                        className="flex w-full items-center gap-2.5 px-3 py-2 text-sm text-text-secondary hover:bg-surface-hover hover:text-accent-red transition-colors"
                      >
                        <LogOut className="h-4 w-4" />
                        {t("dash.sign_out")}
                      </button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </header>

        {/* Content */}
        <main className="flex-1 overflow-y-auto p-4 md:p-6">{children}</main>
      </div>

      {/* AI Chat Assistant */}
      <ChatWidget />
    </div>
  );
}
