"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";
import { Monitor, Rocket, X, Zap } from "lucide-react";
import { toast } from "sonner";
import { useLocale } from "@/lib/locale";
import { useDesktopRuntime } from "@/lib/desktop/runtime";
import { useHoldInteraction } from "@/hooks/useHoldInteraction";
import { useArmedIdleTimeout } from "@/hooks/useArmedIdleTimeout";
import { LaunchInstanceModal } from "@/components/instances/launch-instance-modal";
import { cn } from "@/lib/utils";
import { DeployServerlessModal } from "./deploy-serverless-modal";

interface MobileDeployActionProps {
  canWrite: boolean;
  serverlessEnabled: boolean;
}

type DeployTrack = "instance" | "serverless";

const SWIPE_THRESHOLD_PX = 48;
const ARM_TAP_GRACE_MS = 400;
const ARMED_IDLE_MS = 45_000;
const TOAST_COOLDOWN_MS = 2_500;

function safeVibrate(pattern: number | number[]) {
  try {
    navigator.vibrate?.(pattern);
  } catch {
    /* noop */
  }
}

export function MobileDeployAction({ canWrite, serverlessEnabled }: MobileDeployActionProps) {
  const { t } = useLocale();
  const router = useRouter();
  const pathname = usePathname();
  const { state: desktopState } = useDesktopRuntime();
  const [deployModalOpen, setDeployModalOpen] = useState(false);
  const [launchModalOpen, setLaunchModalOpen] = useState(false);
  const [selectedTrack, setSelectedTrack] = useState<DeployTrack>("instance");
  const [liveMessage, setLiveMessage] = useState("");
  const blockClickUntilRef = useRef(0);
  const lastToastAtRef = useRef(0);
  const modalOpenRef = useRef(false);
  const swipeOriginRef = useRef<{ x: number; y: number } | null>(null);
  const didSwipeRef = useRef(false);

  const announce = useCallback((message: string) => {
    setLiveMessage(message);
  }, []);

  const { progress, isHolding, isArmed, disarm, bind: holdBind } = useHoldInteraction({
    enabled: selectedTrack === "serverless" ? serverlessEnabled : true,
    onArmed: () => {
      blockClickUntilRef.current = Date.now() + ARM_TAP_GRACE_MS;
      safeVibrate([8, 40, 12]);
      announce(
        selectedTrack === "serverless"
          ? t("dash.mobile.deploy_armed")
          : t("dash.mobile.launch_armed"),
      );
    },
  });

  const armedTrack: DeployTrack | null = isArmed ? selectedTrack : null;

  const resetArmed = useCallback(() => {
    disarm();
    blockClickUntilRef.current = 0;
    swipeOriginRef.current = null;
    didSwipeRef.current = false;
  }, [disarm]);

  const resetArmedRef = useRef(resetArmed);
  resetArmedRef.current = resetArmed;

  useArmedIdleTimeout(isArmed && !modalOpenRef.current, () => {
    resetArmed();
    toast.message(t("dash.mobile.action_disarmed_timeout"), { duration: 2200 });
  }, ARMED_IDLE_MS);

  useEffect(() => {
    modalOpenRef.current = deployModalOpen || launchModalOpen;
  }, [deployModalOpen, launchModalOpen]);

  useEffect(() => {
    resetArmedRef.current();
    setDeployModalOpen(false);
    setLaunchModalOpen(false);
  }, [pathname]);

  useEffect(() => {
    if (!isArmed || modalOpenRef.current) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        resetArmed();
        announce(t("dash.mobile.action_disarmed"));
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [isArmed, resetArmed, announce, t]);

  const switchTrack = useCallback((track: DeployTrack) => {
    if (track === "serverless" && !serverlessEnabled) return;
    setSelectedTrack(track);
    resetArmed();
    safeVibrate(6);
    announce(track === "serverless" ? t("dash.mobile.track_serverless") : t("dash.mobile.track_instance"));
  }, [announce, resetArmed, serverlessEnabled, t]);

  const showHintToast = useCallback(() => {
    const now = Date.now();
    if (now - lastToastAtRef.current < TOAST_COOLDOWN_MS) return;
    lastToastAtRef.current = now;
    toast.message(t("dash.mobile.action_hint"), { duration: 2400 });
  }, [t]);

  const handleClick = useCallback((e: React.MouseEvent<HTMLButtonElement>) => {
    if (didSwipeRef.current) {
      didSwipeRef.current = false;
      e.preventDefault();
      return;
    }
    if (isHolding) {
      e.preventDefault();
      return;
    }
    if (Date.now() < blockClickUntilRef.current) {
      e.preventDefault();
      return;
    }
    if (modalOpenRef.current) return;

    if (armedTrack === "serverless") {
      setLaunchModalOpen(false);
      setDeployModalOpen(true);
      return;
    }

    if (armedTrack === "instance") {
      setDeployModalOpen(false);
      setLaunchModalOpen(true);
      return;
    }

    showHintToast();
  }, [armedTrack, isHolding, showHintToast]);

  const onPointerDown = useCallback((e: React.PointerEvent<HTMLButtonElement>) => {
    swipeOriginRef.current = { x: e.clientX, y: e.clientY };
    didSwipeRef.current = false;
    if (isArmed) return;
    holdBind.onPointerDown(e);
  }, [holdBind, isArmed]);

  const onPointerMove = useCallback((e: React.PointerEvent<HTMLButtonElement>) => {
    if (!swipeOriginRef.current || isArmed) {
      holdBind.onPointerMove(e);
      return;
    }
    const dx = e.clientX - swipeOriginRef.current.x;
    const dy = e.clientY - swipeOriginRef.current.y;
    if (Math.abs(dx) > SWIPE_THRESHOLD_PX && Math.abs(dx) > Math.abs(dy) * 1.2) {
      didSwipeRef.current = true;
      if (dx < 0 && selectedTrack === "serverless") {
        switchTrack("instance");
      } else if (dx > 0 && selectedTrack === "instance" && serverlessEnabled) {
        switchTrack("serverless");
      }
      swipeOriginRef.current = null;
      holdBind.onPointerCancel(e);
      return;
    }
    holdBind.onPointerMove(e);
  }, [holdBind, isArmed, selectedTrack, serverlessEnabled, switchTrack]);

  const onPointerUp = useCallback((e: React.PointerEvent<HTMLButtonElement>) => {
    swipeOriginRef.current = null;
    holdBind.onPointerUp(e);
  }, [holdBind]);

  const handleDeployModalClose = useCallback(() => {
    setDeployModalOpen(false);
    resetArmed();
  }, [resetArmed]);

  const handleLaunchModalClose = useCallback(() => {
    setLaunchModalOpen(false);
    resetArmed();
  }, [resetArmed]);

  if (!canWrite) return null;

  const showOnMobile = desktopState.isStandalonePwa;
  const modalOpen = deployModalOpen || launchModalOpen;
  const trackAccent = selectedTrack === "serverless" ? "violet" : "cyan";
  const armedAccent = armedTrack === "serverless" ? "violet" : armedTrack === "instance" ? "cyan" : null;

  return (
    <>
      <div aria-live="polite" aria-atomic="true" className="sr-only">
        {liveMessage}
      </div>

      <div
        className={cn(
          "pointer-events-none fixed inset-x-0 z-[48] flex justify-center",
          showOnMobile
            ? "bottom-0 pb-[max(1.25rem,env(safe-area-inset-bottom))]"
            : "bottom-0 pb-[max(5.5rem,calc(1.25rem+4rem))] max-md:pb-[max(5.5rem,calc(1.25rem+4rem))]",
          showOnMobile ? "flex" : "hidden max-md:flex",
          modalOpen && "opacity-0 pointer-events-none",
        )}
      >
        <div className="pointer-events-auto relative flex flex-col items-center gap-2.5">
          {isArmed && !modalOpen && (
            <button
              type="button"
              onClick={resetArmed}
              className="rounded-full border border-border/70 bg-surface/95 px-3 py-1 text-xs font-medium text-text-muted shadow-md backdrop-blur-sm hover:text-text-primary"
              aria-label={t("dash.mobile.action_disarm_label")}
            >
              <span className="inline-flex items-center gap-1">
                <X className="h-3 w-3" />
                {t("dash.mobile.action_disarm")}
              </span>
            </button>
          )}

          <div className="flex items-center gap-1 rounded-full border border-border/60 bg-surface/90 p-1 shadow-lg backdrop-blur-md">
            <button
              type="button"
              onClick={() => switchTrack("instance")}
              className={cn(
                "rounded-full px-3 py-1 text-[11px] font-semibold transition-all",
                selectedTrack === "instance"
                  ? "bg-accent-cyan/20 text-accent-cyan shadow-[0_0_12px_rgba(34,211,238,0.25)]"
                  : "text-text-muted hover:text-text-secondary",
              )}
            >
              {t("dash.mobile.track_instance")}
            </button>
            {serverlessEnabled && (
              <button
                type="button"
                onClick={() => switchTrack("serverless")}
                className={cn(
                  "rounded-full px-3 py-1 text-[11px] font-semibold transition-all",
                  selectedTrack === "serverless"
                    ? "bg-accent-violet/20 text-accent-violet shadow-[0_0_12px_rgba(139,92,246,0.25)]"
                    : "text-text-muted hover:text-text-secondary",
                )}
              >
                {t("dash.mobile.track_serverless")}
              </button>
            )}
          </div>

          <p className="text-[10px] font-medium uppercase tracking-[0.14em] text-text-muted/80">
            {t("dash.mobile.track_switch_hint")}
          </p>

          <motion.button
            type="button"
            aria-pressed={isArmed}
            aria-expanded={modalOpen}
            aria-disabled={modalOpen}
            disabled={modalOpen}
            aria-label={
              armedTrack === "serverless"
                ? t("dash.mobile.deploy_armed_label")
                : armedTrack === "instance"
                  ? t("dash.mobile.launch_armed_label")
                  : t("dash.mobile.action_idle_label")
            }
            className={cn(
              "relative flex h-[4.25rem] min-w-[min(92vw,22rem)] items-center justify-center gap-3 overflow-hidden rounded-2xl px-8 text-base font-semibold shadow-2xl transition-shadow",
              armedAccent === "violet"
                ? "text-white shadow-[0_0_48px_rgba(139,92,246,0.5)]"
                : armedAccent === "cyan"
                  ? "text-navy shadow-[0_0_48px_rgba(34,211,238,0.45)]"
                  : "border border-border/70 bg-surface text-text-primary shadow-[0_16px_48px_rgba(0,0,0,0.4)]",
            )}
            animate={
              armedAccent
                ? { scale: [1, 1.02, 1] }
                : isHolding
                  ? { scale: 1.01 }
                  : { scale: 1 }
            }
            transition={
              armedAccent
                ? { duration: 1.8, repeat: Infinity, ease: "easeInOut" }
                : { duration: 0.2 }
            }
            onClick={handleClick}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerCancel={holdBind.onPointerCancel}
            onLostPointerCapture={holdBind.onLostPointerCapture}
            style={{ touchAction: "none", userSelect: "none", WebkitUserSelect: "none" }}
          >
            {/* Ambient gradient mesh */}
            <span
              className={cn(
                "pointer-events-none absolute inset-0 opacity-80",
                armedAccent === "violet"
                  ? "bg-gradient-to-r from-accent-violet via-fuchsia-500 to-indigo-500 animate-pulse"
                  : armedAccent === "cyan"
                    ? "bg-gradient-to-r from-accent-cyan via-sky-400 to-emerald-400 animate-pulse"
                    : trackAccent === "violet"
                      ? "bg-gradient-to-br from-accent-violet/10 via-surface to-accent-cyan/5"
                      : "bg-gradient-to-br from-accent-cyan/10 via-surface to-accent-violet/5",
              )}
              aria-hidden
            />

            {/* Hold progress — bottom edge fill, no ring */}
            {!isArmed && (isHolding || progress > 0) && (
              <span
                className={cn(
                  "pointer-events-none absolute inset-x-0 bottom-0 transition-[height] duration-75",
                  selectedTrack === "serverless"
                    ? "bg-gradient-to-t from-accent-violet/70 to-accent-violet/10"
                    : "bg-gradient-to-t from-accent-cyan/70 to-accent-cyan/10",
                )}
                style={{ height: `${Math.max(8, progress * 100)}%` }}
                aria-hidden
              />
            )}

            <AnimatePresence mode="wait" initial={false}>
              {armedTrack === "serverless" ? (
                <motion.span
                  key="armed-serverless"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  transition={{ duration: 0.2 }}
                  className="relative z-10 flex items-center gap-2.5"
                >
                  <Rocket className="h-5 w-5" aria-hidden />
                  {t("dash.mobile.tap_to_open")}
                </motion.span>
              ) : armedTrack === "instance" ? (
                <motion.span
                  key="armed-instance"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  transition={{ duration: 0.2 }}
                  className="relative z-10 flex items-center gap-2.5"
                >
                  <Monitor className="h-5 w-5" aria-hidden />
                  {t("dash.mobile.tap_to_open")}
                </motion.span>
              ) : (
                <motion.span
                  key="idle"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  transition={{ duration: 0.2 }}
                  className="relative z-10 flex items-center gap-2.5 text-center leading-tight"
                >
                  {isHolding ? (
                    <Zap className="h-5 w-5 shrink-0 animate-pulse text-accent-cyan" aria-hidden />
                  ) : selectedTrack === "serverless" ? (
                    <Rocket className="h-5 w-5 shrink-0 text-accent-violet" aria-hidden />
                  ) : (
                    <Monitor className="h-5 w-5 shrink-0 text-accent-cyan" aria-hidden />
                  )}
                  <span className="max-w-[15rem]">
                    {isHolding
                      ? t("dash.mobile.deploy_holding")
                      : serverlessEnabled
                        ? t("dash.mobile.action_idle")
                        : t("dash.mobile.action_idle_instances")}
                  </span>
                </motion.span>
              )}
            </AnimatePresence>
          </motion.button>
        </div>
      </div>

      <DeployServerlessModal
        open={deployModalOpen}
        onClose={handleDeployModalClose}
        canWrite={canWrite}
      />

      <LaunchInstanceModal
        open={launchModalOpen}
        onClose={handleLaunchModalClose}
        onLaunched={(jobId) => {
          setLaunchModalOpen(false);
          resetArmed();
          router.push(`/dashboard/instances/${jobId}`);
        }}
      />
    </>
  );
}