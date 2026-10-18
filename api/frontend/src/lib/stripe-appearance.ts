import type { Appearance } from "@stripe/stripe-js";

/**
 * Single source of truth for the Xcelsior dark dashboard's Stripe brand palette.
 * Both the Elements appearance (checkout / payment method / deposit) and the
 * Connect embedded appearance (onboarding / account management / payouts) are
 * derived from this so every embedded Stripe surface matches — no more off-brand
 * accents drifting between components.
 */
const PALETTE = {
  primary: "#00d4ff", // brand cyan
  primaryContrast: "#04121a", // readable text on the cyan button
  surface: "#0f172a", // component background
  inputBg: "#0b1220", // input / tab background
  border: "#1e293b",
  text: "#e2e8f0",
  mutedText: "#94a3b8",
  danger: "#dc2626",
  success: "#10b981",
  font: "ui-sans-serif, system-ui, -apple-system, sans-serif",
  radius: "10px",
  focusRing: "0 0 0 2px rgba(0, 212, 255, 0.25)",
} as const;

/** Shared embedded Stripe Elements appearance, matches Xcelsior dark dashboard theme. */
export const STRIPE_APPEARANCE: Appearance = {
  theme: "night",
  labels: "floating",
  variables: {
    colorPrimary: PALETTE.primary,
    colorBackground: PALETTE.surface,
    colorText: PALETTE.text,
    colorDanger: PALETTE.danger,
    colorSuccess: PALETTE.success,
    fontFamily: PALETTE.font,
    fontSizeBase: "14px",
    spacingUnit: "4px",
    borderRadius: PALETTE.radius,
    focusBoxShadow: PALETTE.focusRing,
  },
  rules: {
    ".Input": {
      backgroundColor: PALETTE.inputBg,
      border: `1px solid ${PALETTE.border}`,
      boxShadow: "none",
      padding: "12px 14px",
    },
    ".Input:focus": {
      border: "1px solid rgba(0, 212, 255, 0.45)",
    },
    ".Label": {
      color: PALETTE.mutedText,
      fontSize: "12px",
      fontWeight: "500",
    },
    ".Tab": {
      backgroundColor: PALETTE.inputBg,
      border: `1px solid ${PALETTE.border}`,
    },
    ".Tab--selected": {
      backgroundColor: "rgba(0, 212, 255, 0.08)",
      border: "1px solid rgba(0, 212, 255, 0.35)",
      color: PALETTE.text,
    },
    ".Block": {
      backgroundColor: "transparent",
      boxShadow: "none",
    },
  },
};

/**
 * Connect embedded components (account-onboarding, account-management, payouts,
 * notification-banner) take a *different* appearance shape than Elements. Derive
 * it from the same PALETTE so the payouts portal matches the rest of the app
 * instead of shipping its own ad-hoc (emerald) accent.
 */
export const STRIPE_CONNECT_APPEARANCE = {
  overlays: "dialog" as const,
  variables: {
    colorPrimary: PALETTE.primary,
    colorBackground: PALETTE.surface,
    colorText: PALETTE.text,
    colorSecondaryText: PALETTE.mutedText,
    colorBorder: PALETTE.border,
    colorDanger: PALETTE.danger,
    buttonPrimaryColorBackground: PALETTE.primary,
    buttonPrimaryColorText: PALETTE.primaryContrast,
    borderRadius: PALETTE.radius,
    fontFamily: PALETTE.font,
    spacingUnit: "4px",
  },
};

/**
 * Card + Link checkout UI. Stripe under the hood.
 * Hides Apple Pay / Google Pay / BNPL tabs — PayPal stays a separate app path.
 */
export const STRIPE_PAYMENT_ELEMENT_OPTIONS = {
  layout: "tabs" as const,
  paymentMethodOrder: ["card", "link"],
  wallets: {
    applePay: "never" as const,
    googlePay: "never" as const,
    link: "auto" as const,
  },
};

export function getStripeElementsOptions(clientSecret?: string) {
  return {
    appearance: STRIPE_APPEARANCE,
    ...(clientSecret ? { clientSecret } : {}),
  } as const;
}
