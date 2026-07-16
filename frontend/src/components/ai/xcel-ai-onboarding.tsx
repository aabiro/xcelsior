"use client";

import { useCallback, useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { createPortal } from "react-dom";
import { ArrowRight, ChevronLeft, Rocket, Server, Sparkles, Terminal, Wallet, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useLocale } from "@/lib/locale";

const STORAGE_KEY = "xcelsior-ai-onboarding-v1";

type Slide = {
  id: string;
  image: string;
  titleKey: string;
  bodyKey: string;
  bullets?: { icon: React.ElementType; textKey: string }[];
};

const SLIDES: Slide[] = [
  {
    id: "hero",
    image: "/ai-onboarding/hero.svg",
    titleKey: "ai.onboarding.hero_title",
    bodyKey: "ai.onboarding.hero_body",
  },
  {
    id: "actions",
    image: "/ai-onboarding/actions.svg",
    titleKey: "ai.onboarding.actions_title",
    bodyKey: "ai.onboarding.actions_body",
    bullets: [
      { icon: Rocket, textKey: "ai.onboarding.bullet_instances" },
      { icon: Server, textKey: "ai.onboarding.bullet_hosts" },
      { icon: Wallet, textKey: "ai.onboarding.bullet_billing" },
      { icon: Terminal, textKey: "ai.onboarding.bullet_settings" },
    ],
  },
  {
    id: "api",
    image: "/ai-onboarding/api.svg",
    titleKey: "ai.onboarding.api_title",
    bodyKey: "ai.onboarding.api_body",
  },
  {
    id: "ready",
    image: "/ai-onboarding/ready.svg",
    titleKey: "ai.onboarding.ready_title",
    bodyKey: "ai.onboarding.ready_body",
  },
];

export function hasSeenAiOnboarding(): boolean {
  if (typeof window === "undefined") return true;
  try {
    return localStorage.getItem(STORAGE_KEY) === "1";
  } catch {
    return true;
  }
}

export function markAiOnboardingSeen(): void {
  try {
    localStorage.setItem(STORAGE_KEY, "1");
  } catch {
    /* noop */
  }
  void fetch("/api/users/me/preferences", {
    method: "PUT",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ preferences: { ai_onboarding_seen: true } }),
  }).catch(() => {});
}

export function XcelAiOnboarding({
  open,
  onClose,
  onComplete,
}: {
  open: boolean;
  onClose: () => void;
  onComplete?: () => void;
}) {
  const { t } = useLocale();
  const [step, setStep] = useState(0);

  const finish = useCallback(() => {
    setStep(0);
    markAiOnboardingSeen();
    onComplete?.();
    onClose();
  }, [onClose, onComplete]);

  if (!open) return null;

  const slide = SLIDES[step];
  const isLast = step === SLIDES.length - 1;

  return createPortal(
    <div className="fixed inset-0 z-[300] flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-navy/75 backdrop-blur-[3px]" onClick={finish} aria-hidden />
      <motion.div
        initial={{ opacity: 0, scale: 0.96, y: 12 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.96, y: 12 }}
        className="relative z-10 flex max-h-[calc(100dvh-2rem)] w-full max-w-lg flex-col overflow-hidden rounded-2xl border border-border/60 bg-surface shadow-2xl shadow-black/50"
      >
        <button
          type="button"
          onClick={finish}
          className="absolute right-4 top-4 z-20 rounded-lg p-1.5 text-text-muted hover:bg-surface-hover hover:text-text-primary transition-colors"
          aria-label="Close"
        >
          <X className="h-4 w-4" />
        </button>

        <div className="relative min-h-0 overflow-y-auto">
          <AnimatePresence mode="wait">
            <motion.div
              key={slide.id}
              initial={{ opacity: 0, x: 24 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -24 }}
              transition={{ duration: 0.22 }}
            >
              <div className="relative flex h-44 w-full items-center justify-center overflow-hidden bg-navy/70 p-4 sm:h-52 sm:p-5">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={slide.image}
                  alt=""
                  className="relative z-0 max-h-full max-w-full object-contain"
                />
                <div className="pointer-events-none absolute inset-x-0 bottom-0 z-10 h-12 bg-gradient-to-t from-surface to-transparent" />
              </div>
              <div className="px-6 pb-6 pt-4">
                <div className="flex items-center gap-2 mb-2">
                  <Sparkles className="h-4 w-4 text-accent-cyan" />
                  <span className="text-[10px] font-semibold uppercase tracking-widest text-accent-cyan/80">
                    {t("ai.onboarding.badge")}
                  </span>
                </div>
                <h2 className="text-xl font-bold tracking-tight text-text-primary">{t(slide.titleKey)}</h2>
                <p className="mt-2 text-sm leading-relaxed text-text-secondary">{t(slide.bodyKey)}</p>
                {slide.bullets && (
                  <ul className="mt-4 grid grid-cols-2 gap-2">
                    {slide.bullets.map(({ icon: Icon, textKey }) => (
                      <li
                        key={textKey}
                        className="flex items-center gap-2 rounded-lg border border-border/50 bg-navy/40 px-2.5 py-2 text-xs text-text-secondary"
                      >
                        <Icon className="h-3.5 w-3.5 shrink-0 text-accent-cyan" />
                        {t(textKey)}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </motion.div>
          </AnimatePresence>
        </div>

        <div className="flex items-center justify-between border-t border-border/60 px-6 py-4">
          <div className="flex items-center gap-1.5">
            {SLIDES.map((s, i) => (
              <span
                key={s.id}
                className={cn(
                  "h-1.5 rounded-full transition-all duration-200",
                  i === step ? "w-6 bg-accent-cyan" : "w-1.5 bg-border",
                )}
              />
            ))}
          </div>
          <div className="flex items-center gap-2">
            {step > 0 && (
              <Button type="button" variant="ghost" size="sm" onClick={() => setStep((s) => s - 1)}>
                <ChevronLeft className="h-4 w-4" />
                {t("common.back")}
              </Button>
            )}
            {isLast ? (
              <Button type="button" size="sm" className="bg-accent-cyan text-navy hover:bg-accent-cyan/90" onClick={finish}>
                {t("ai.onboarding.cta")}
                <ArrowRight className="h-4 w-4" />
              </Button>
            ) : (
              <Button type="button" size="sm" className="bg-accent-cyan text-navy hover:bg-accent-cyan/90" onClick={() => setStep((s) => s + 1)}>
                {t("common.next")}
                <ArrowRight className="h-4 w-4" />
              </Button>
            )}
          </div>
        </div>
      </motion.div>
    </div>,
    document.body,
  );
}

export function useAiOnboardingGate() {
  const [show, setShow] = useState(false);

  useEffect(() => {
    if (!hasSeenAiOnboarding()) {
      const id = requestAnimationFrame(() => setShow(true));
      return () => cancelAnimationFrame(id);
    }
  }, []);

  return { showOnboarding: show, setShowOnboarding: setShow, dismissOnboarding: () => setShow(false) };
}
