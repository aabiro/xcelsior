"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard, Server, Monitor, Activity, CreditCard,
  Store, DollarSign, ShieldCheck, Star, FileCheck,
  BarChart3, Package, Calendar, Settings, Users, ChevronLeft,
  ChevronRight, LogOut, Shield, Cpu, Menu, X, Key, ChevronDown,
  Zap, HardDrive, TrendingUp, BookOpen, Rocket, CheckCircle2, Circle,
  ExternalLink, HelpCircle, Sparkles,
} from "lucide-react";
import { useAuth } from "@/lib/auth";
import { useLocale } from "@/lib/locale";
import { cn } from "@/lib/utils";
import { AnimatePresence, motion } from "framer-motion";
import { Breadcrumb } from "@/components/ui/breadcrumb";
import { ThemeToggle } from "@/components/ui/theme-toggle";
import { LocaleToggle } from "@/components/ui/locale-toggle";
import { ChatWidget } from "@/components/ChatWidget";
import { AiPanel } from "@/components/AiPanel";
import { NotificationBell } from "@/components/NotificationBell";

const AI_PANEL_KEY = "xcelsior-ai-panel-open";

const navItems: { href: string; key: string; icon: typeof LayoutDashboard; roles?: string[] }[] = [
  { href: "/dashboard/ai", key: "dash.ai", icon: Sparkles },
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
  const [gearOpen, setGearOpen] = useState(false);
  const [onboardingOpen, setOnboardingOpen] = useState(false);
  const [aiPanelOpen, setAiPanelOpen] = useState(false);
  const profileRef = useRef<HTMLDivElement>(null);
  const gearRef = useRef<HTMLDivElement>(null);
  const pathname = usePathname();
  const { user, loading: authLoading, logout } = useAuth();
  const { t } = useLocale();

  // Restore AI panel state from localStorage
  useEffect(() => {
    try {
      const stored = localStorage.getItem(AI_PANEL_KEY);
      if (stored === "true") setAiPanelOpen(true);
    } catch { /* SSR */ }
  }, []);

  const toggleAiPanel = useCallback(() => {
    setAiPanelOpen((prev) => {
      const next = !prev;
      try { localStorage.setItem(AI_PANEL_KEY, String(next)); } catch { /* noop */ }
      return next;
    });
  }, []);

  const closeAiPanel = useCallback(() => {
    setAiPanelOpen(false);
    try { localStorage.setItem(AI_PANEL_KEY, "false"); } catch { /* noop */ }
  }, []);

  // Redirect to login if not authenticated
  useEffect(() => {
    if (!authLoading && !user) {
      window.location.href = `/login?redirect=${encodeURIComponent(pathname)}`;
    }
  }, [authLoading, user, pathname]);

  // Close mobile drawer on route change
  useEffect(() => { setMobileOpen(false); setProfileOpen(false); setGearOpen(false); setOnboardingOpen(false); }, [pathname]);

  // Close profile dropdown on click outside
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (profileRef.current && !profileRef.current.contains(e.target as Node)) {
        setProfileOpen(false);
      }
      if (gearRef.current && !gearRef.current.contains(e.target as Node)) {
        setGearOpen(false);
        setOnboardingOpen(false);
      }
    }
    if (profileOpen || gearOpen) document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [profileOpen, gearOpen]);

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
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/xcelsior-logo-wordmark-iconbg.svg" alt="Xcelsior" className={cn("hidden dark:block", collapsed && !mobile ? "h-9 w-9 object-left object-cover overflow-hidden" : "h-11")} style={collapsed && !mobile ? { clipPath: "inset(0 74% 0 0)" } : undefined} />
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/xcelsior-logo-wordmark-iconbg-light.svg" alt="Xcelsior" className={cn("block dark:hidden", collapsed && !mobile ? "h-9 w-9 object-left object-cover overflow-hidden" : "h-11")} style={collapsed && !mobile ? { clipPath: "inset(0 74% 0 0)" } : undefined} />
          {(mobile || !collapsed) && (
            <span className="rounded bg-accent-cyan/10 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-accent-cyan">
              Beta
            </span>
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

      {/* Gear Popout + Collapse (desktop only) */}
      {!mobile && (
        <div className="border-t border-border p-2 space-y-0.5">
          <div className="relative" ref={gearRef}>
            <button
              onClick={() => setGearOpen(!gearOpen)}
              className={cn(
                "flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors",
                gearOpen
                  ? "bg-accent-cyan/8 text-accent-cyan"
                  : "text-text-secondary hover:bg-surface-hover hover:text-text-primary"
              )}
              title={collapsed ? t("gear.title") : undefined}
            >
              <HelpCircle className="h-4 w-4 shrink-0" />
              {!collapsed && <span>{t("gear.title")}</span>}
            </button>

            {/* Gear Popout Panel */}
            <AnimatePresence>
              {gearOpen && (
                <motion.div
                  initial={{ opacity: 0, y: 8, scale: 0.96 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 8, scale: 0.96 }}
                  transition={{ duration: 0.15 }}
                  className="absolute bottom-full left-0 mb-2 w-72 rounded-xl border border-border/60 bg-surface shadow-xl z-50"
                >
                  {/* Quick links */}
                  <div className="p-2">
                    <Link
                      href="/dashboard/settings"
                      className="flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm text-text-secondary hover:bg-surface-hover hover:text-text-primary transition-colors"
                      onClick={() => setGearOpen(false)}
                    >
                      <Settings className="h-4 w-4" />
                      {t("gear.settings")}
                    </Link>
                    <a
                      href="https://docs.xcelsior.ca"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm text-text-secondary hover:bg-surface-hover hover:text-text-primary transition-colors"
                      onClick={() => setGearOpen(false)}
                    >
                      <BookOpen className="h-4 w-4" />
                      {t("gear.docs")}
                      <ExternalLink className="h-3 w-3 ml-auto text-text-muted" />
                    </a>
                    <button
                      onClick={() => setOnboardingOpen(!onboardingOpen)}
                      className={cn(
                        "flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-sm transition-colors",
                        onboardingOpen
                          ? "bg-accent-cyan/8 text-accent-cyan"
                          : "text-text-secondary hover:bg-surface-hover hover:text-text-primary"
                      )}
                    >
                      <Rocket className="h-4 w-4" />
                      {t("gear.onboarding")}
                      <ChevronRight className="h-3.5 w-3.5 ml-auto" />
                    </button>
                  </div>

                  {/* Onboarding sub-popout */}
                  <AnimatePresence>
                    {onboardingOpen && (
                      <motion.div
                        initial={{ opacity: 0, x: -8, scale: 0.96 }}
                        animate={{ opacity: 1, x: 0, scale: 1 }}
                        exit={{ opacity: 0, x: -8, scale: 0.96 }}
                        transition={{ duration: 0.15 }}
                        className="absolute left-full bottom-0 ml-2 w-72 rounded-xl border border-border/60 bg-surface shadow-xl z-50 overflow-hidden"
                      >
                        <GearOnboarding t={t} onNavigate={() => { setGearOpen(false); setOnboardingOpen(false); }} user={user} />
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="flex w-full items-center justify-center rounded-lg px-3 py-2 text-text-muted hover:bg-surface-hover hover:text-text-primary"
            title={collapsed ? t("gear.expand") : t("gear.collapse")}
          >
            {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
          </button>
        </div>
      )}
      {/* Settings + Docs (mobile drawer) */}
      {mobile && (
        <div className="border-t border-border p-2 space-y-0.5">
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
            <span>{t("gear.settings")}</span>
          </Link>
          <a
            href="https://docs.xcelsior.ca"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm text-text-secondary hover:bg-surface-hover hover:text-text-primary transition-colors"
          >
            <BookOpen className="h-4 w-4 shrink-0" />
            <span>{t("gear.docs")}</span>
            <ExternalLink className="h-3 w-3 ml-auto text-text-muted" />
          </a>
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
                    className="absolute right-0 top-full mt-1 w-56 rounded-xl border border-border/60 bg-surface shadow-xl z-50 overflow-hidden"
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
                        href="/dashboard/settings#team"
                        className="flex items-center gap-2.5 px-3 py-2 text-sm text-text-secondary hover:bg-surface-hover hover:text-text-primary transition-colors"
                        onClick={() => setProfileOpen(false)}
                      >
                        <Users className="h-4 w-4" />
                        {t("dash.team") || "Team"}
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

      {/* AI Context Panel (right side) */}
      <AnimatePresence>
        {aiPanelOpen && (
          <motion.aside
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 384, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ type: "spring", damping: 30, stiffness: 280, mass: 0.8 }}
            className="hidden md:flex flex-col border-l border-border/30 bg-background/95 backdrop-blur-md ai-panel-border overflow-hidden"
          >
            <AiPanel onClose={closeAiPanel} />
          </motion.aside>
        )}
      </AnimatePresence>

      {/* AI Chat Assistant */}
      <ChatWidget onOpenAiPanel={toggleAiPanel} />
    </div>
  );
}

/* ── Onboarding Checklist Component ─────────────────────────────────── */

const ONBOARDING_STEPS = [
  { key: "profile", labelKey: "gear.step_profile", descKey: "gear.step_profile_desc", href: "/dashboard/settings" },
  { key: "jurisdiction", labelKey: "gear.step_jurisdiction", descKey: "gear.step_jurisdiction_desc", href: "/dashboard/settings" },
  { key: "api_key", labelKey: "gear.step_api_key", descKey: "gear.step_api_key_desc", href: "/dashboard/settings#api-keys" },
  { key: "browse", labelKey: "gear.step_browse", descKey: "gear.step_browse_desc", href: "/dashboard/marketplace" },
  { key: "instance", labelKey: "gear.step_instance", descKey: "gear.step_instance_desc", href: "/dashboard/instances/new" },
] as const;

const STORAGE_KEY = "xcelsior-onboarding";

function useOnboardingState() {
  const [completed, setCompleted] = useState<Record<string, boolean>>({});
  const loadedRef = useRef(false);

  // Load from server, fall back to localStorage
  useEffect(() => {
    if (loadedRef.current) return;
    loadedRef.current = true;
    fetch("/api/users/me/preferences", { credentials: "include" })
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        const serverOnboarding = data?.preferences?.onboarding;
        if (serverOnboarding && typeof serverOnboarding === "object") {
          setCompleted(serverOnboarding);
          // Sync to localStorage as cache
          try { localStorage.setItem(STORAGE_KEY, JSON.stringify(serverOnboarding)); } catch {}
        } else {
          // Fall back to localStorage for migration
          try {
            const raw = localStorage.getItem(STORAGE_KEY);
            if (raw) {
              const parsed = JSON.parse(raw);
              setCompleted(parsed);
              // Migrate localStorage data to server
              fetch("/api/users/me/preferences", {
                method: "PUT",
                credentials: "include",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ preferences: { onboarding: parsed } }),
              }).catch(() => {});
            }
          } catch {}
        }
      })
      .catch(() => {
        // Offline fallback to localStorage
        try {
          const raw = localStorage.getItem(STORAGE_KEY);
          if (raw) setCompleted(JSON.parse(raw));
        } catch {}
      });
  }, []);

  const toggle = (key: string) => {
    setCompleted((prev) => {
      const next = { ...prev, [key]: !prev[key] };
      // Persist to localStorage as cache
      try { localStorage.setItem(STORAGE_KEY, JSON.stringify(next)); } catch {}
      // Persist to server
      fetch("/api/users/me/preferences", {
        method: "PUT",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ preferences: { onboarding: next } }),
      }).catch(() => {});
      return next;
    });
  };

  return { completed, toggle };
}

function GearOnboarding({
  t,
  onNavigate,
  user,
}: {
  t: (key: string, vars?: Record<string, string | number>) => string;
  onNavigate: () => void;
  user: { name?: string; email?: string; role?: string } | null;
}) {
  const { completed, toggle } = useOnboardingState();
  const doneCount = ONBOARDING_STEPS.filter((s) => completed[s.key]).length;
  const allDone = doneCount === ONBOARDING_STEPS.length;

  return (
    <div className="p-3">
      <div className="flex items-center gap-2 mb-2">
        <Rocket className="h-4 w-4 text-accent-gold" />
        <span className="text-sm font-semibold">{t("gear.onboarding")}</span>
      </div>
      <p className="text-xs text-text-muted mb-3 leading-relaxed">{t("gear.onboarding_desc")}</p>

      {/* Progress bar */}
      <div className="mb-3">
        <div className="flex items-center justify-between text-xs text-text-muted mb-1">
          <span>{allDone ? t("gear.all_done") : t("gear.progress", { done: doneCount, total: ONBOARDING_STEPS.length })}</span>
          <span className="font-mono">{Math.round((doneCount / ONBOARDING_STEPS.length) * 100)}%</span>
        </div>
        <div className="h-1.5 w-full rounded-full bg-surface-hover overflow-hidden">
          <div
            className="h-full rounded-full bg-gradient-to-r from-accent-cyan to-accent-violet transition-all duration-300"
            style={{ width: `${(doneCount / ONBOARDING_STEPS.length) * 100}%` }}
          />
        </div>
      </div>

      {/* Steps */}
      <div className="space-y-1">
        {ONBOARDING_STEPS.map((step) => {
          const done = !!completed[step.key];
          return (
            <div key={step.key} className="flex items-start gap-2 group">
              <button
                onClick={() => toggle(step.key)}
                className="mt-0.5 shrink-0"
                aria-label={done ? "Mark incomplete" : "Mark complete"}
              >
                {done ? (
                  <CheckCircle2 className="h-4 w-4 text-emerald" />
                ) : (
                  <Circle className="h-4 w-4 text-text-muted group-hover:text-accent-cyan transition-colors" />
                )}
              </button>
              <div className="flex-1 min-w-0">
                <Link
                  href={step.href}
                  onClick={onNavigate}
                  className={cn(
                    "text-sm leading-tight transition-colors hover:text-accent-cyan",
                    done ? "text-text-muted line-through" : "text-text-primary"
                  )}
                >
                  {t(step.labelKey)}
                </Link>
                <p className="text-[11px] text-text-muted leading-snug mt-0.5">{t(step.descKey)}</p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
