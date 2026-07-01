"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
  Settings, Users, ChevronLeft, ChevronRight, LogOut, Menu, X, Key, ChevronDown,
  BookOpen, Rocket, ExternalLink, HelpCircle, Clock, MessageCircle,
} from "lucide-react";
import { DashboardNav } from "@/components/nav/dashboard-nav";
import { UserAvatar } from "@/components/user/user-avatar";
import { useAuth } from "@/lib/auth";
import { BRAND_ASSETS } from "@/lib/brand-assets";
import { getTeamContext, formatTeamRoleLabel } from "@/lib/team-context";
import { useLocale } from "@/lib/locale";
import { useTheme } from "@/lib/theme";
import { cn } from "@/lib/utils";
import { AnimatePresence, motion } from "framer-motion";
import { Breadcrumb } from "@/components/ui/breadcrumb";
import { ThemeToggle } from "@/components/ui/theme-toggle";
import { LocaleToggle } from "@/components/ui/locale-toggle";
import { ChatWidget } from "@/components/ChatWidget";
import { AiPanel } from "@/components/AiPanel";
import { AiSparkIcon } from "@/components/ui/ai-spark-icon";
import { GlobalLaunchModal } from "@/components/instances/global-launch-modal";
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
  const { theme } = useTheme();
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

  // Full-screen gate only on first session probe (not every tab change).
  if (authLoading && !user) {
    return (
      <div className="dashboard-shell" data-theme={theme}>
        <div className="dashboard-shell-loading">
          <div className="dashboard-shell-loading-spinner animate-spin" />
        </div>
      </div>
    );
  }
  if (!user) {
    return null;
  }

  const canAccessRole = (requiredRole: string) => (
    requiredRole === "admin" ? (!!user.is_admin || user.role === "admin") : user.role === requiredRole
  );

  const sidebarContent = (mobile: boolean) => (
    <>
      {/* Logo */}
      <div className="dashboard-site-sidebar-header flex items-center justify-between">
        <Link href="/dashboard" className="flex items-center overflow-visible pr-2">
          <div className="relative flex min-w-0 items-center" style={{ width: collapsed && !mobile ? 36 : undefined }}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={BRAND_ASSETS.iconGradientTight}
              alt="Xcelsior"
              className={cn(
                "absolute left-1/2 top-1/2 h-8 w-8 -translate-x-1/2 -translate-y-1/2 object-contain transition-all duration-300 ease-in-out",
                collapsed && !mobile
                  ? "opacity-100 scale-100"
                  : "pointer-events-none opacity-0 scale-75"
              )}
            />
            <div
              className={cn(
                "flex min-w-0 items-center gap-2 transition-all duration-300 ease-in-out",
                collapsed && !mobile ? "pointer-events-none w-0 overflow-hidden opacity-0 scale-95" : "opacity-100 scale-100"
              )}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={BRAND_ASSETS.lockupLight}
                alt="Xcelsior"
                className="hidden h-auto w-[140px] shrink-0 dark:block"
              />
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={BRAND_ASSETS.lockupDark}
                alt="Xcelsior"
                className="block h-auto w-[140px] shrink-0 dark:hidden"
              />
              <span className="dashboard-site-beta shrink-0 rounded-full px-1.5 py-0.5 text-[11px] font-semibold uppercase">
                Beta
              </span>
            </div>
          </div>
        </Link>
        {mobile && (
          <button onClick={() => setMobileOpen(false)} className="dashboard-site-close rounded-xl p-2" aria-label="Close menu">
            <X className="h-5 w-5" />
          </button>
        )}
      </div>

      <div className="dashboard-site-nav">
        <DashboardNav
          collapsed={collapsed}
          mobile={mobile}
          showServerless={showServerless}
          canAccessRole={canAccessRole}
          t={t}
          onNavigate={mobile ? () => setMobileOpen(false) : undefined}
        />
      </div>

      {/* Gear Popout + Collapse (desktop only) */}
      {!mobile && (
        <div className="dashboard-site-sidebar-footer p-2 space-y-0.5">
          <div className="relative" ref={gearRef}>
            <button
              onClick={() => setGearOpen(!gearOpen)}
              className={cn(
                "dashboard-site-sidebutton flex w-full items-center gap-3 rounded-2xl px-3 py-2 text-base",
                gearOpen && "dashboard-site-sidebutton-active"
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
                  className="dashboard-site-popout absolute bottom-full left-0 z-50 mb-2 w-72 rounded-[22px]"
                >
                  {/* Quick links */}
                  <div className="p-2">
                    <Link
                      href="/dashboard/settings"
                      className="dashboard-site-popout-link flex items-center gap-2.5 rounded-2xl px-3 py-2 text-sm"
                      onClick={() => setGearOpen(false)}
                    >
                      <Settings className="h-4 w-4" />
                      {t("gear.settings")}
                    </Link>
                    <a
                      href="https://docs.xcelsior.ca"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="dashboard-site-popout-link flex items-center gap-2.5 rounded-2xl px-3 py-2 text-sm"
                      onClick={() => setGearOpen(false)}
                    >
                      <BookOpen className="h-4 w-4" />
                      {t("gear.docs")}
                      <ExternalLink className="h-3 w-3 ml-auto text-text-muted" />
                    </a>
                    <button
                      onClick={() => { setSupportPopoutOpen(!supportPopoutOpen); setOnboardingOpen(false); }}
                      className={cn(
                        "dashboard-site-popout-link flex w-full items-center gap-2.5 rounded-2xl px-3 py-2 text-sm",
                        supportPopoutOpen && "dashboard-site-popout-link-active"
                      )}
                    >
                      <MessageCircle className="h-4 w-4" />
                      {t("gear.support")}
                      <ChevronRight className="h-3.5 w-3.5 ml-auto" />
                    </button>
                    <button
                      onClick={() => { setOnboardingOpen(!onboardingOpen); setSupportPopoutOpen(false); }}
                      className={cn(
                        "dashboard-site-popout-link flex w-full items-center gap-2.5 rounded-2xl px-3 py-2 text-sm",
                        onboardingOpen && "dashboard-site-popout-link-active"
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
                        className="dashboard-site-subpanel absolute bottom-0 left-full z-50 ml-2 w-72 overflow-hidden rounded-[22px]"
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
                        className="dashboard-site-subpanel absolute bottom-0 left-full z-50 ml-2 h-[500px] w-[360px] overflow-hidden rounded-[22px]"
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
            className="dashboard-site-collapse flex w-full items-center justify-center rounded-2xl px-3 py-2"
            title={collapsed ? t("gear.expand") : t("gear.collapse")}
          >
            {collapsed ? <ChevronRight className="h-5 w-5" /> : <ChevronLeft className="h-5 w-5" />}
          </button>
        </div>
      )}
      {/* Settings + Docs + Getting Started (mobile drawer) */}
      {mobile && (
        <div className="dashboard-site-sidebar-footer p-2 space-y-0.5">
          <button
            type="button"
            onClick={() => setMobileOnboardingOpen((o) => !o)}
            className={cn(
              "dashboard-site-mobile-link flex w-full items-center gap-3 rounded-2xl px-3 py-2 text-base",
              mobileOnboardingOpen && "dashboard-site-sidebutton-active",
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
              "dashboard-site-mobile-link flex items-center gap-3 rounded-2xl px-3 py-2 text-base",
              pathname.startsWith("/dashboard/settings") && "dashboard-site-sidebutton-active nav-active"
            )}
          >
            <Settings className="h-5 w-5 shrink-0" />
            <span>{t("gear.settings")}</span>
          </Link>
          <a
            href="https://docs.xcelsior.ca"
            target="_blank"
            rel="noopener noreferrer"
            className="dashboard-site-mobile-link flex items-center gap-3 rounded-2xl px-3 py-2 text-base"
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
    <div className="dashboard-shell" data-theme={theme}>
      <div className={cn("dashboard-shell-frame flex h-screen overflow-hidden", desktopMode && "desktop-shell-root")}>
        {/* Desktop Sidebar */}
        <aside
          className={cn(
            "dashboard-site-sidebar hidden md:flex flex-col transition-all duration-200",
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
                className="dashboard-site-mobile-backdrop fixed inset-0 z-40 md:hidden"
                onClick={() => setMobileOpen(false)}
              />
              <motion.aside
                initial={{ x: -280 }}
                animate={{ x: 0 }}
                exit={{ x: -280 }}
                transition={{ type: "spring", damping: 25, stiffness: 300 }}
                className="dashboard-site-mobile-drawer fixed inset-y-0 left-0 z-50 flex w-[280px] flex-col md:hidden"
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
          <header className={cn("dashboard-site-topbar glass relative", desktopMode && "desktop-topbar")}>
            <div className="brand-line absolute bottom-0 left-0 right-0" />
            <div className="dashboard-site-topbar-inner">
              {/* Mobile menu button */}
              <button
                className="dashboard-site-icon-button flex h-11 w-11 items-center justify-center rounded-full md:hidden"
                onClick={() => setMobileOpen(true)}
                aria-label="Open menu"
              >
                <Menu className="h-6 w-6" />
              </button>
              <div className="dashboard-site-crumbs hidden min-w-0 md:block">
                <Breadcrumb />
              </div>
              <DesktopStatusStrip className="dashboard-site-status hidden xl:flex" />
              <div className={cn("dashboard-site-actions flex items-center", desktopMode && "desktop-topbar-actions")}>
                <LocaleToggle className="dashboard-site-pill-control" />
                <ThemeToggle className="dashboard-site-pill-control" />
                <div className="h-6 w-px bg-[var(--line)] hidden sm:block" />
                <div className="dashboard-site-control">
                  <NotificationBell />
                </div>
                <TeamSwitcher compact className="dashboard-site-team-control" />
                <div className="dashboard-site-control">
                  <CreditsButton />
                </div>
                <div className="h-7 w-px bg-[var(--line)] hidden sm:block" />
                <div className="relative" ref={profileRef}>
                  <button
                    onClick={() => setProfileOpen(!profileOpen)}
                    className="dashboard-site-profile-trigger flex items-center gap-2 rounded-full px-3 py-2"
                  >
                    <UserAvatar user={user} size="md" />
                    {user && (
                      <div className="hidden text-left sm:block">
                        <p className="text-base font-medium leading-none text-[var(--text)]">{user.name || user.email}</p>
                        <p className="text-sm text-[var(--text-4)]">
                          {team.isTeamMember
                            ? `${team.teamName || t("dash.team")} · ${formatTeamRoleLabel(team.teamRole)}`
                            : user.is_admin
                              ? (user.role && user.role !== "admin" ? `Admin · ${user.role}` : "Admin")
                              : user.role || "user"}
                        </p>
                      </div>
                    )}
                    <ChevronDown className={cn("hidden h-4 w-4 text-[var(--text-4)] transition-transform sm:block", profileOpen && "rotate-180")} />
                  </button>
                  {/* Profile dropdown */}
                  <AnimatePresence>
                    {profileOpen && (
                      <motion.div
                        initial={{ opacity: 0, y: -4 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -4 }}
                        transition={{ duration: 0.15 }}
                        className="dashboard-site-popout absolute right-0 top-full z-50 mt-2 w-56 overflow-hidden rounded-[22px]"
                      >
                        <div className="dashboard-site-popout-section flex items-center gap-3 border-b px-3 py-3">
                          <UserAvatar user={user} size="sm" />
                          <div className="min-w-0">
                            <p className="truncate text-sm font-medium text-[var(--text)]">{user?.name || user?.email}</p>
                            <p className="truncate text-xs text-[var(--text-4)]">{user?.email}</p>
                          </div>
                        </div>
                        <div className="py-1">
                          <Link
                            href="/dashboard/settings"
                            className="dashboard-site-popout-link flex items-center gap-2.5 px-3 py-2 text-sm"
                            onClick={() => setProfileOpen(false)}
                          >
                            <Settings className="h-4 w-4" />
                            {t("dash.settings")}
                          </Link>
                          <Link
                            href="/dashboard/settings#team"
                            className="dashboard-site-popout-link flex items-center gap-2.5 px-3 py-2 text-sm"
                            onClick={() => setProfileOpen(false)}
                          >
                            <Users className="h-4 w-4" />
                            {t("dash.team") || "Team"}
                          </Link>
                          <Link
                            href="/dashboard/settings#api-keys"
                            className="dashboard-site-popout-link flex items-center gap-2.5 px-3 py-2 text-sm"
                            onClick={() => setProfileOpen(false)}
                          >
                            <Key className="h-4 w-4" />
                            {t("dash.settings.api_keys") || "API Keys"}
                          </Link>
                        </div>
                        <div className="dashboard-site-popout-section border-t py-1">
                          <button
                            onClick={() => { setProfileOpen(false); void logout(); }}
                            className="dashboard-site-popout-link flex w-full items-center gap-2.5 px-3 py-2 text-sm hover:text-[var(--coral)]"
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
            </div>
          </header>

          {/* Content */}
          <main className={cn("dashboard-site-main flex-1 overflow-y-auto", desktopMode && "desktop-main-surface")}>
            <div className="dashboard-site-main-inner">{children}</div>
          </main>
        </div>

        {/* Global Launch Instance modal — opened in place from any "Launch" button */}
        <GlobalLaunchModal />

        {/* AI Context Panel (right side) */}
        <AnimatePresence>
          {aiPanelOpen && (
            <motion.aside
              initial={{ width: 0, opacity: 0 }}
              animate={{ width: 384, opacity: 1 }}
              exit={{ width: 0, opacity: 0 }}
              transition={{ type: "spring", damping: 30, stiffness: 280, mass: 0.8 }}
              className="dashboard-site-ai-panel ai-panel-border hidden overflow-hidden md:flex md:flex-col"
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
        <div className="dashboard-site-ai-rail hidden w-16 shrink-0 flex-col items-center justify-end py-3 md:flex">
          <button
            onClick={toggleAiPanel}
            className={cn(
              "dashboard-site-ai-toggle flex h-[60px] w-[60px] items-center justify-center rounded-[22px] p-[3px] transition-all duration-200",
              aiPanelOpen && "dashboard-site-ai-toggle-active text-white"
            )}
            title={aiPanelOpen ? t("ai.close_panel") : t("ai.open_panel")}
          >
            <AiSparkIcon className={cn("h-[54px] w-[54px] transition-transform duration-200", aiPanelOpen && "rotate-12")} />
          </button>
        </div>
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
        className="dashboard-site-banner"
      >
        <div className="dashboard-site-banner-content flex items-center justify-center gap-3 text-sm">
          <Clock className="h-4 w-4 shrink-0" />
          <span>Your session will expire soon due to inactivity</span>
          <button
            onClick={continueSession}
            className="dashboard-site-banner-action ml-1 rounded-full px-3 py-1 text-xs font-medium"
          >
            Continue Session
          </button>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
