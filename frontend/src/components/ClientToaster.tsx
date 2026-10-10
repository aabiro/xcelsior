"use client";

import { useSyncExternalStore, type CSSProperties } from "react";
import dynamic from "next/dynamic";
import { usePathname } from "next/navigation";
import { AlertTriangle, CheckCircle2, Info, Loader2, XCircle } from "lucide-react";
import { useTheme } from "@/lib/theme";

const Toaster = dynamic(() => import("sonner").then((m) => ({ default: m.Toaster })), {
  ssr: false,
});

// Sonner's own look is switched off (`unstyled`); the whole card is drawn by
// the .xcelsior-toast rules in globals.css, in both themes.
const ICONS = {
  success: <CheckCircle2 aria-hidden />,
  info: <Info aria-hidden />,
  warning: <AlertTriangle aria-hidden />,
  error: <XCircle aria-hidden />,
  loading: <Loader2 aria-hidden className="animate-spin" />,
};

const PHONE = "(max-width: 640px)";

function subscribePhone(onChange: () => void) {
  const query = window.matchMedia(PHONE);
  query.addEventListener("change", onChange);
  return () => query.removeEventListener("change", onChange);
}

export function ClientToaster() {
  const { theme } = useTheme();
  // A phone-height dialog puts its primary action at the bottom, which is
  // exactly where a bottom stack lands: "tick the box first" covered the box.
  // On a phone the stack drops from the top instead.
  const phone = useSyncExternalStore(subscribePhone, () => window.matchMedia(PHONE).matches, () => false);
  // The dashboard's AI rail is a 64px column on the right edge; keep the
  // stack clear of it instead of covering its launcher.
  const inDashboard = usePathname()?.startsWith("/dashboard") ?? false;
  return (
    <Toaster
      position={phone ? "top-center" : "bottom-right"}
      theme={theme}
      closeButton
      expand={false}
      gap={12}
      visibleToasts={3}
      icons={ICONS}
      offset={{ bottom: 24, right: inDashboard ? 88 : 24 }}
      mobileOffset={{ top: 12, left: 12, right: 12 }}
      style={{ "--width": "384px" } as CSSProperties}
      toastOptions={{
        duration: 4000,
        unstyled: true,
        classNames: {
          toast: "xcelsior-toast",
          title: "xcelsior-toast-title",
          description: "xcelsior-toast-description",
          icon: "xcelsior-toast-icon",
          closeButton: "xcelsior-toast-dismiss",
        },
      }}
    />
  );
}
