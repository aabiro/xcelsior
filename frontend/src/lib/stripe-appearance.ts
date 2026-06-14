import type { Appearance } from "@stripe/stripe-js";

/** Shared embedded Stripe Elements appearance — matches Xcelsior dark dashboard theme. */
export const STRIPE_APPEARANCE: Appearance = {
  theme: "night",
  labels: "floating",
  variables: {
    colorPrimary: "#00d4ff",
    colorBackground: "#0f172a",
    colorText: "#e2e8f0",
    colorDanger: "#dc2626",
    colorSuccess: "#10b981",
    fontFamily: "ui-sans-serif, system-ui, -apple-system, sans-serif",
    fontSizeBase: "14px",
    spacingUnit: "4px",
    borderRadius: "10px",
    focusBoxShadow: "0 0 0 2px rgba(0, 212, 255, 0.25)",
  },
  rules: {
    ".Input": {
      backgroundColor: "#0b1220",
      border: "1px solid #1e293b",
      boxShadow: "none",
      padding: "12px 14px",
    },
    ".Input:focus": {
      border: "1px solid rgba(0, 212, 255, 0.45)",
    },
    ".Label": {
      color: "#94a3b8",
      fontSize: "12px",
      fontWeight: "500",
    },
    ".Tab": {
      backgroundColor: "#0b1220",
      border: "1px solid #1e293b",
    },
    ".Tab--selected": {
      backgroundColor: "rgba(0, 212, 255, 0.08)",
      border: "1px solid rgba(0, 212, 255, 0.35)",
      color: "#e2e8f0",
    },
    ".Block": {
      backgroundColor: "transparent",
      boxShadow: "none",
    },
  },
};

/** Card-only — excludes Pix, Klarna, wallets, and other APMs from Payment Element. */
export const STRIPE_PAYMENT_ELEMENT_OPTIONS = {
  layout: "tabs" as const,
  paymentMethodOrder: ["card"],
  wallets: {
    applePay: "never" as const,
    googlePay: "never" as const,
  },
};

export function getStripeElementsOptions(clientSecret?: string) {
  return {
    appearance: STRIPE_APPEARANCE,
    ...(clientSecret ? { clientSecret } : {}),
  } as const;
}