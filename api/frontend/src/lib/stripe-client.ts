import { loadStripe, type Stripe } from "@stripe/stripe-js";

let stripePromise: Promise<Stripe | null> | null = null;

/** Resolve publishable key, prefers explicit env, falls back to live/sandbox pair. */
export function getStripePublishableKey(): string | undefined {
  return (
    process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY ||
    process.env.NEXT_PUBLIC_STRIPE_LIVE_PUBLISHABLE_KEY ||
    process.env.NEXT_PUBLIC_STRIPE_SANDBOX_PUBLISHABLE_KEY ||
    undefined
  );
}

export function getStripePromise(): Promise<Stripe | null> | null {
  const key = getStripePublishableKey();
  if (!key) return null;
  if (!stripePromise) {
    stripePromise = loadStripe(key);
  }
  return stripePromise;
}