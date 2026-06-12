"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
  LayoutDashboard, Server, Monitor, Activity, CreditCard,
  Store, DollarSign, ShieldCheck, Star, FileCheck,
  BarChart3, Package, Calendar, Settings, Users, ChevronLeft,
  ChevronRight, LogOut, Shield, Cpu, Menu, X, Key, ChevronDown,
  Zap, HardDrive, TrendingUp, BookOpen, Rocket,
  ExternalLink, HelpCircle, Sparkles, Clock, MessageCircle, Layers,
} from "lucide-react";
import { useAuth } from "@/lib/auth";
import { getTeamContext, formatTeamRoleLabel } from "@/lib/team-context";
import { useLocale } from "@/lib/locale";
import { cn } from "@/lib/utils";
import { AnimatePresence, motion } from "framer-motion";
import { Breadcrumb } from "@/components/ui/breadcrumb";
import { Badge } from "@/components/ui/badge";
import { ThemeToggle } from "@/components/ui/theme-toggle";
import { LocaleToggle } from "@/components/ui/locale-toggle";
import { ChatWidget } from "@/components/ChatWidget";
import { AiPanel } from "@/components/AiPanel";
import { DesktopStatusStrip } from "@/components/DesktopStatusStrip";
import { NotificationBell } from "@/components/NotificationBell";
import { CreditsButton } from "@/components/CreditsButton";
import { TeamSwitcher } from "@/components/team/team-switcher";
import { useDesktopRuntime } from "@/lib/desktop/runtime";
import { MobileDeployAction } from "@/components/mobile/mobile-deploy-action";
import { GearOnboarding } from "@/components/onboarding/gear-onboarding";
import * as api from "@/lib/api";

const AI_PANEL_KEY = "xcelsior-ai-panel-open";
const SIDEBAR_COLLAPSED_KEY = "xcelsior-sidebar-collapsed";

function readStoredFlag(key: string, match = "true"): boolean {
  if (typeof window === "undefined") return false;
  try {
    return localStorage.getItem(key) === match;
  } catch {
    return false;
  }
}

const navItems: { href: string; key: string; icon: typeof LayoutDashboard; roles?: string[]; badge?: string }[] = [
  { href: "/dashboard/ai", key: "dash.ai", icon: Sparkles, badge: "New" },
  { href: "/dashboard", key: "dash.overview", icon: LayoutDashboard },
  { href: "/dashboard/hosts", key: "dash.hosts", icon: Server },
  { href: "/dashboard/instances", key: "dash.instances", icon: Monitor },
  { href: "/dashboard/templates", key: "dash.templates", icon: Layers },
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
  { href: "/dashboard/hpc", key: "dash.hpc", icon: Cpu, roles: ["admin"] },
  { href: "/dashboard/events", key: "dash.events", icon: Calendar },
  { href: "/dashboard/admin", key: "dash.admin", icon: Users, roles: ["admin"] },
];

export function DashboardShell({ children }: { children: React.ReactNode }) {
  const [collapsed, setCollapsed] = useState(() => readStoredFlag(SIDEBAR_COLLAPSED_KEY, "1"));
  const [mobileOpen, setMobileOpen] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);
  const [gearOpen, setGearOpen] = useState(false);
  const [onboardingOpen, setOnboardingOpen] = useState(false);
  const [mobileOnboardingOpen, setMobileOnboardingOpen] = useState(false);
  const [supportPopoutOpen, setSupportPopoutOpen] = useState(false);
  const [aiPanelOpen, setAiPanelOpen] = useState(() => readStoredFlag(AI_PANEL_KEY));
  const [serverlessEnabled, setServerlessEnabled] = useState(false);
  const profileRef = useRef<HTMLDivElement>(null);
  const gearRef = useRef<HTMLDivElement>(null);
  const pathname = usePathname();
  const router = useRouter();
  const { user, loading: authLoading, logout } = useAuth();
  const { t } = useLocale();
  const team = getTeamContext(user);
  const canWriteServerless = team.canWriteInstances;
  const { state: desktopState, openControlCenter } = useDesktopRuntime();
  const desktopMode = desktopState.isNativeDesktop || desktopState.isStandalonePwa;

  useEffect(() => {
    try {
      localStorage.setItem(SIDEBAR_COLLAPSED_KEY, collapsed ? "1" : "0");
    } catch { /* noop */ }
  }, [collapsed]);

  const toggleAiPanel = useCallback(() => {
    setAiPanelOpen((prev) => {
      const next = !prev;
      try { localStorage.setItem(AI_PANEL_KEY, String(next)); } catch { /* noop */ }
      if (next && pathname === "/dashboard/ai") {
        router.push("/dashboard");
      }
      return next;
    });
  }, [pathname, router]);

  const closeAiPanel = useCallback(() => {
    setAiPanelOpen(false);
    try { localStorage.setItem(AI_PANEL_KEY, "false"); } catch { /* noop */ }
    window.dispatchEvent(new CustomEvent("xcelsior-close-ai-panel"));
  }, []);

  // Listen for custom events from AI full-page to open/close panel
  useEffect(() => {
    const openHandler = () => {
      setAiPanelOpen(true);
      try { localStorage.setItem(AI_PANEL_KEY, "true"); } catch { /* noop */ }
    };
    const closeHandler = () => {
      setAiPanelOpen(false);
      try { localStorage.setItem(AI_PANEL_KEY, "false"); } catch { /* noop */ }
    };
    window.addEventListener("xcelsior-open-ai-panel", openHandler);
    window.addEventListener("xcelsior-close-ai-panel", closeHandler);
    return () => {
      window.removeEventListener("xcelsior-open-ai-panel", openHandler);
      window.removeEventListener("xcelsior-close-ai-panel", closeHandler);
    };
  }, []);

  // Redirect to login if not authenticated
  useEffect(() => {
    if (!authLoading && !user) {
      window.location.href = `/login?redirect=${encodeURIComponent(pathname)}`;
    }
  }, [authLoading, user, pathname]);

  useEffect(() => {
    if (!user) return;
    let cancelled = false;
    void api.getServerlessEnabled()
      .then((res) => { if (!cancelled) setServerlessEnabled(!!res.enabled); })
      .catch(() => { if (!cancelled) setServerlessEnabled(false); });
    return () => { cancelled = true; };
  }, [user]);

  const showServerless = user ? serverlessEnabled : false;

  // Close mobile drawer on route change
  useEffect(() => {
    const id = requestAnimationFrame(() => {
      setMobileOpen(false);
      setProfileOpen(false);
      setGearOpen(false);
      setOnboardingOpen(false);
    });
    return () => cancelAnimationFrame(id);
  }, [pathname]);

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

  useEffect(() => {
    if (!desktopMode) return;

    const handleKeydown = (event: KeyboardEvent) => {
      const modifier = event.metaKey || event.ctrlKey;
      if (!modifier) return;

      const key = event.key.toLowerCase();
      if (event.shiftKey && key === "m") {
        event.preventDefault();
        router.push("/dashboard/marketplace");
      } else if (event.shiftKey && key === "i") {
        event.preventDefault();
        router.push("/dashboard/instances");
      } else if (event.shiftKey && key === "n") {
        event.preventDefault();
        router.push("/dashboard/notifications");
      } else if (event.shiftKey && key === "d" && desktopState.isNativeDesktop) {
        event.preventDefault();
        void openControlCenter("/desktop");
      } else if (!event.shiftKey && key === "b") {
        event.preventDefault();
        setCollapsed((current) => !current);
      } else if (!event.shiftKey && key === "k") {
        event.preventDefault();
        toggleAiPanel();
      }
    };

    window.addEventListener("keydown", handleKeydown);
    return () => window.removeEventListener("keydown", handleKeydown);
  }, [desktopMode, desktopState.isNativeDesktop, openControlCenter, router, toggleAiPanel]);

  // Show loading state while checking auth
  if (authLoading || !user) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-accent-red border-t-transparent" />
      </div>
    );
  }

  const canAccessRole = (requiredRole: string) => (
    requiredRole === "admin" ? (!!user.is_admin || user.role === "admin") : user.role === requiredRole
  );

  const sidebarContent = (mobile: boolean) => (
    <>
      {/* Logo */}
      <div className="flex h-[72px] items-center border-b border-border/60 px-4 justify-between">
        <Link href="/dashboard" className="flex items-center overflow-visible pr-2">
          <div className="relative shrink-0" style={{ width: collapsed && !mobile ? 40 : undefined }}>
            {/* Rounded icon — visible when collapsed */}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src="/xcelsior-logo-rounded.svg"
              alt="Xcelsior"
              className={cn(
                "h-[40px] w-[40px] transition-all duration-300 ease-in-out absolute top-1/2 left-0",
                collapsed && !mobile
                  ? "opacity-100 scale-100 -translate-y-1/2"
                  : "opacity-0 scale-75 -translate-y-1/2 pointer-events-none"
              )}
            />
            <div
              className={cn(
                "flex items-center gap-2 transition-all duration-300 ease-in-out",
                collapsed && !mobile ? "pointer-events-none w-0 overflow-hidden opacity-0 scale-95" : "opacity-100 scale-100"
              )}
            >
              {/* Wordmark — visible when expanded */}
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src="/xcelsior-logo-wordmark-iconbg.svg"
                alt="Xcelsior"
                className="hidden h-[47px] w-auto shrink-0 dark:block"
              />
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src="/xcelsior-logo-wordmark-iconbg-light.svg"
                alt="Xcelsior"
                className="block h-[47px] w-auto shrink-0 dark:hidden"
              />
              <span className="shrink-0 rounded-full bg-accent-cyan/8 px-1.5 py-0.5 text-[11px] font-semibold uppercase tracking-widest text-accent-cyan/70">Beta</span>
            </div>
          </div>
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
          .filter((item) => !item.roles || item.roles.some(canAccessRole))
          .filter((item) => item.href !== "/dashboard/inference" || showServerless)
          .map((item) => {
          const active = pathname === item.href || (item.href !== "/dashboard" && pathname.startsWith(item.href));
          const label = t(item.key);
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2 text-base transition-colors",
                active
                  ? "bg-accent-cyan/8 text-accent-cyan nav-active"
                  : "text-text-secondary hover:bg-surface-hover hover:text-text-primary"
              )}
              title={!mobile && collapsed ? label : undefined}
            >
              <item.icon className={`h-5 w-5 shrink-0 ${active ? 'drop-shadow-[0_0_4px_rgba(0,212,255,0.5)]' : ''}`} />
              {(mobile || !collapsed) && (
                <div className="flex items-center gap-2">
                  <span>{label}</span>
                  {item.badge && (
                    <span className="shrink-0 rounded-full bg-accent-cyan/8 px-1.5 py-0.5 text-[11px] font-semibold uppercase tracking-widest text-accent-cyan/70">
                      {item.badge}
                    </span>
                  )}
                </div>
              )}
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
                "flex w-full items-center gap-3 rounded-lg px-3 py-2 text-base transition-colors",
                gearOpen
                  ? "bg-accent-cyan/8 text-accent-cyan"
                  : "text-text-secondary hover:bg-surface-hover hover:text-text-primary"
              )}
              title={collapsed ? t("gear.title") : undefined}
            >
              <HelpCircle className="h-5 w-5 shrink-0" />
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
                      onClick={() => { setSupportPopoutOpen(!supportPopoutOpen); setOnboardingOpen(false); }}
                      className={cn(
                        "flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-sm transition-colors",
                        supportPopoutOpen
                          ? "bg-accent-cyan/8 text-accent-cyan"
                          : "text-text-secondary hover:bg-surface-hover hover:text-text-primary"
                      )}
                    >
                      <MessageCircle className="h-4 w-4" />
                      {t("gear.support")}
                      <ChevronRight className="h-3.5 w-3.5 ml-auto" />
                    </button>
                    <button
                      onClick={() => { setOnboardingOpen(!onboardingOpen); setSupportPopoutOpen(false); }}
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
                        <GearOnboarding t={t} onNavigate={() => { setGearOpen(false); setOnboardingOpen(false); }} user={user} pathname={pathname} />
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {/* Support chat sub-popout */}
                  <AnimatePresence>
                    {supportPopoutOpen && (
                      <motion.div
                        initial={{ opacity: 0, x: -8, scale: 0.96 }}
                        animate={{ opacity: 1, x: 0, scale: 1 }}
                        exit={{ opacity: 0, x: -8, scale: 0.96 }}
                        transition={{ duration: 0.15 }}
                        className="absolute left-full bottom-0 ml-2 w-[360px] h-[500px] rounded-xl border border-border/60 bg-surface shadow-xl z-50 overflow-hidden"
                      >
                        <ChatWidget
                          showFab={false}
                          externalOpen={true}
                          onClose={() => { setSupportPopoutOpen(false); }}
                          embedded
                        />
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
            {collapsed ? <ChevronRight className="h-5 w-5" /> : <ChevronLeft className="h-5 w-5" />}
          </button>
        </div>
      )}
      {/* Settings + Docs + Getting Started (mobile drawer) */}
      {mobile && (
        <div className="border-t border-border p-2 space-y-0.5">
          <button
            type="button"
            onClick={() => setMobileOnboardingOpen((o) => !o)}
            className={cn(
              "flex w-full items-center gap-3 rounded-lg px-3 py-2 text-base transition-colors",
              mobileOnboardingOpen
                ? "bg-accent-cyan/8 text-accent-cyan"
                : "text-text-secondary hover:bg-surface-hover hover:text-text-primary",
            )}
            aria-expanded={mobileOnboardingOpen}
          >
            <Rocket className="h-5 w-5 shrink-0" />
            <span>{t("gear.onboarding")}</span>
            <ChevronRight
              className={cn(
                "h-4 w-4 ml-auto text-text-muted transition-transform",
                mobileOnboardingOpen && "rotate-90",
              )}
            />
          </button>
          <AnimatePresence initial={false}>
            {mobileOnboardingOpen && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="overflow-hidden"
              >
                <GearOnboarding
                  t={t}
                  user={user}
                  pathname={pathname}
                  onNavigate={() => {
                    setMobileOpen(false);
                    setMobileOnboardingOpen(false);
                  }}
                  className="px-1 pb-1"
                />
              </motion.div>
            )}
          </AnimatePresence>
          <Link
            href="/dashboard/settings"
            className={cn(
              "flex items-center gap-3 rounded-lg px-3 py-2 text-base transition-colors",
              pathname.startsWith("/dashboard/settings")
                ? "bg-accent-cyan/8 text-accent-cyan nav-active"
                : "text-text-secondary hover:bg-surface-hover hover:text-text-primary"
            )}
          >
            <Settings className="h-5 w-5 shrink-0" />
            <span>{t("gear.settings")}</span>
          </Link>
          <a
            href="https://docs.xcelsior.ca"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-3 rounded-lg px-3 py-2 text-base text-text-secondary hover:bg-surface-hover hover:text-text-primary transition-colors"
          >
            <BookOpen className="h-5 w-5 shrink-0" />
            <span>{t("gear.docs")}</span>
            <ExternalLink className="h-3.5 w-3.5 ml-auto text-text-muted" />
          </a>
        </div>
      )}
    </>
  );

  return (
    <div className={cn("flex h-screen overflow-hidden bg-navy", desktopMode && "desktop-shell-root")}>
      {/* Desktop Sidebar */}
      <aside
        className={cn(
          "hidden md:flex flex-col border-r border-border/60 glass transition-all duration-200",
          desktopMode && "desktop-sidebar-surface",
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
        {/* Session expiry warning — above topbar */}
        <SessionExpiryBanner />

        {/* Topbar */}
        <header className={cn("flex h-[72px] items-center justify-between glass px-4 md:px-6 relative", desktopMode && "desktop-topbar")}>
          <div className="brand-line absolute bottom-0 left-0 right-0" />
          {/* Mobile menu button */}
          <button
            className="md:hidden flex items-center justify-center h-10 w-10 rounded-lg text-text-secondary hover:bg-surface-hover hover:text-text-primary"
            onClick={() => setMobileOpen(true)}
            aria-label="Open menu"
          >
            <Menu className="h-6 w-6" />
          </button>
          <div className="hidden md:block min-w-0">
            <Breadcrumb />
          </div>
          <DesktopStatusStrip className="hidden xl:flex" />
          <div className={cn("flex items-center gap-4", desktopMode && "desktop-topbar-actions")}>
            <LocaleToggle />
            <ThemeToggle />
            <div className="h-6 w-px bg-border hidden sm:block" />
            <NotificationBell />
            <TeamSwitcher compact />
            <CreditsButton />
            <div className="h-7 w-px bg-border hidden sm:block" />
            <div className="relative" ref={profileRef}>
              <button
                onClick={() => setProfileOpen(!profileOpen)}
                className="flex items-center gap-2 rounded-lg px-2 py-1.5 hover:bg-surface-hover transition-colors"
              >
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-accent-cyan/15 text-lg font-medium text-accent-cyan ring-1 ring-accent-cyan/20">
                  {user?.name?.[0]?.toUpperCase() || user?.email?.[0]?.toUpperCase() || "?"}
                </div>
                {user && (
                  <div className="hidden sm:block text-left">
                    <p className="text-base font-medium leading-none">{user.name || user.email}</p>
                    <p className="text-sm text-text-muted">
                      {team.isTeamMember
                        ? `${team.teamName || t("dash.team")} · ${formatTeamRoleLabel(team.teamRole)}`
                        : user.is_admin
                          ? (user.role && user.role !== "admin" ? `Admin · ${user.role}` : "Admin")
                          : user.role || "user"}
                    </p>
                  </div>
                )}
                <ChevronDown className={cn("h-4 w-4 text-text-muted transition-transform hidden sm:block", profileOpen && "rotate-180")} />
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
        <main className={cn("flex-1 overflow-y-auto p-4 md:p-6", desktopMode && "desktop-main-surface")}>{children}</main>
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

      <MobileDeployAction
        serverlessEnabled={showServerless}
        canWrite={canWriteServerless}
      />

      {/* AI Toggle Rail (persistent right edge) */}
      <div className="hidden md:flex flex-col items-center justify-end border-l border-border/30 bg-surface/50 w-11 py-3 shrink-0">
        <button
          onClick={toggleAiPanel}
          className={cn(
            "flex h-11 w-11 items-center justify-center rounded-lg p-0.5 transition-all duration-200",
            aiPanelOpen
              ? "bg-accent-red text-white shadow-lg shadow-accent-red/25"
              : "text-accent-red hover:bg-accent-red/15"
          )}
          title={aiPanelOpen ? t("ai.close_panel") : t("ai.open_panel")}
        >
          <Sparkles className={cn("h-[47px] w-[47px] transition-transform duration-200", aiPanelOpen && "rotate-12")} />
        </button>
      </div>
    </div>
  );
}

/* ── Session Expiry Warning Banner ──────────────────────────────────── */

function SessionExpiryBanner() {
  const { sessionExpiring, continueSession } = useAuth();
  if (!sessionExpiring) return null;
  return (
    <AnimatePresence>
      <motion.div
        initial={{ height: 0, opacity: 0 }}
        animate={{ height: "auto", opacity: 1 }}
        exit={{ height: 0, opacity: 0 }}
        transition={{ duration: 0.2 }}
        className="flex items-center justify-center gap-3 bg-gradient-to-r from-accent-cyan/12 via-accent-violet/10 to-accent-cyan/12 border-b border-accent-cyan/20 px-4 py-2.5 text-sm text-accent-cyan"
      >
        <Clock className="h-4 w-4 shrink-0" />
        <span>Your session will expire soon due to inactivity</span>
        <button
          onClick={continueSession}
          className="ml-1 rounded-md bg-accent-cyan/15 border border-accent-cyan/30 px-3 py-1 text-xs font-medium text-accent-cyan hover:bg-accent-cyan/25 transition-colors"
        >
          Continue Session
        </button>
      </motion.div>
    </AnimatePresence>
  );
}


