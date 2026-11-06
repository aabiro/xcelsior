"use client";

import { Suspense, useEffect, useState } from "react";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { useAuth } from "@/lib/auth";
import { useTheme } from "@/lib/theme";
import { normalizeAuthRedirectPath, apiFetch, registerProvider, type ProviderAccount } from "@/lib/api";
import { AccountProfileForm } from "@/components/ui/province-selector";
import { ProviderProfileForm } from "@/components/providers/seller-profile-card";
import { StripeConnectEmbedded } from "@/components/providers/stripe-connect-embedded";
import { PayoutRequirements } from "@/components/providers/payout-requirements";
import { Button } from "@/components/ui/button";
import { SITE_ASSETS } from "@/lib/brand-assets";
import { cn } from "@/lib/utils";
import { ArrowLeft, Check, MapPin, Moon, Server, ShieldCheck, Sun, UserRound } from "lucide-react";

function SetupFlow() {
  const { user, loading, refreshUser } = useAuth();
  const { theme, toggleTheme } = useTheme();
  const router = useRouter(); const params = useParams<{ step?: string[] }>(); const search = useSearchParams();
  const requested = params.step?.[0] || "account";
  const step = ["account", "address", "provider", "verification"].includes(requested) ? requested : "account";
  const rawTarget = normalizeAuthRedirectPath(search.get("redirect"), "/dashboard");
  const target = rawTarget.startsWith("/onboarding") ? "/dashboard" : rawTarget;
  const go = (next: string) => router.push(`/onboarding/${next}?redirect=${encodeURIComponent(target)}`);
  const [provider, setProvider] = useState<ProviderAccount | null>(null);
  const [error, setError] = useState(""); const [starting, setStarting] = useState(false);
  useEffect(() => {
    if (!loading && !user) router.replace(`/login?redirect=${encodeURIComponent(`/onboarding/${step}?redirect=${encodeURIComponent(target)}`)}`);
  }, [loading, user, router, step, target]);
  useEffect(() => {
    if (!user || step === "account") return;
    const missing = user.missing_profile_fields || [];
    const accountMissing = ["name", "email", "account_type", "onboarding_intent", "business_name"].some((field) => missing.includes(field));
    const addressMissing = ["country", "province", "address_line1", "city", "postal_code"].some((field) => missing.includes(field));
    const providerStep = step === "provider" || step === "verification";
    const next = accountMissing || (providerStep && user.onboarding_intent === "rent") ? "account"
      : step !== "address" && addressMissing ? "address"
      : step === "verification" && !user.provider_id ? "provider" : null;
    if (next) router.replace(`/onboarding/${next}?redirect=${encodeURIComponent(target)}`);
  }, [user, step, router, target]);
  async function refreshProvider() {
    if (!user?.provider_id) return;
    const result = await apiFetch<{provider: ProviderAccount}>(`/api/providers/${encodeURIComponent(user.provider_id)}`);
    setProvider(result.provider); setError("");
  }
  useEffect(() => {
    if (step !== "verification" || !user?.provider_id) return;
    let active = true;
    apiFetch<{provider: ProviderAccount}>(`/api/providers/${encodeURIComponent(user.provider_id)}`)
      .then((result) => { if (active) setProvider(result.provider); })
      .catch((err) => { if (active) setError(err instanceof Error ? err.message : "Unable to load verification status"); });
    return () => { active = false; };
  }, [step, user?.provider_id]);
  if (loading || !user) return <p role="status" className="p-8">Loading your account…</p>;
  const providerIntent = ["provide", "both"].includes(user.onboarding_intent || "");
  const titles: Record<string,string> = {account:"Tell us how you’ll use Xcelsior",address:"Your address",provider:"Your provider profile",verification:"Provider verification"};
  const flow = [
    { id: "account", title: "Your account", description: "Personal, business, or provider", icon: UserRound },
    { id: "address", title: "Your address", description: "Accurate details, entered once", icon: MapPin },
    ...(providerIntent ? [
      { id: "provider", title: "Provider profile", description: "Seller and tax information", icon: Server },
      { id: "verification", title: "Provider checks", description: "Identity and payout setup", icon: ShieldCheck },
    ] : []),
  ];
  const currentIndex = Math.max(flow.findIndex((item) => item.id === step), 0);
  const missing = user.missing_profile_fields || [];
  const accountComplete = user.profile_complete === true || !["name", "email", "account_type", "onboarding_intent", "business_name"].some((field) => missing.includes(field));
  const addressComplete = user.profile_complete === true || !["country", "province", "address_line1", "city", "postal_code"].some((field) => missing.includes(field));
  return <main className="dashboard-shell onboarding-shell relative isolate min-h-screen bg-navy pb-32 text-text-primary" data-theme={theme}>
    <div aria-hidden className="pointer-events-none absolute inset-0 -z-10" style={{background:"radial-gradient(ellipse at 10% 5%, rgb(14 165 233 / 0.09), transparent 50%), radial-gradient(ellipse at 95% 90%, rgb(124 58 237 / 0.06), transparent 45%)"}} />
    <div className="mx-auto w-full max-w-6xl px-5 py-6 sm:px-8 sm:py-8 lg:px-12">
      <header className="mb-10 flex items-center justify-between gap-4 lg:mb-16">
        <Link href="/" aria-label="Xcelsior home" className="inline-flex rounded-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ice-blue">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={theme === "light" ? SITE_ASSETS.wordmarkDark : SITE_ASSETS.wordmarkLight} alt="Xcelsior" className="h-7 w-auto sm:h-8" />
        </Link>
        <Button type="button" variant="outline" size="sm" onClick={toggleTheme} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}>
          {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </Button>
      </header>
      <div className="grid items-start gap-8 lg:grid-cols-[260px_minmax(0,1fr)] lg:gap-14">
        <aside className="lg:sticky lg:top-10">
          <p className="mb-3 text-xs font-semibold tracking-[0.18em] text-accent-cyan">ACCOUNT SETUP</p>
          <h2 className="text-2xl font-semibold tracking-tight sm:text-3xl">Welcome to Xcelsior.</h2>
          <p className="mt-3 max-w-sm text-sm leading-relaxed text-text-secondary">A few details to get you started. Your setup follows how you plan to use the platform.</p>
          <nav aria-label="Setup progress" className="mt-7 lg:mt-10">
            <ol className="grid grid-cols-2 gap-2 lg:grid-cols-1 lg:gap-3">
              {flow.map((item,index) => {
                const active = step === item.id;
                const done = item.id === "account" ? accountComplete : item.id === "address" ? addressComplete : item.id === "provider" ? provider?.seller_profile_complete === true : false;
                const Icon = item.icon;
                return <li key={item.id} aria-current={active ? "step" : undefined}>
                  <div className={cn("flex items-center gap-3 rounded-xl border p-3 transition-colors", active ? "border-accent-cyan/30 bg-accent-cyan/5" : "border-transparent")}>
                    <span className={cn("flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border", active ? "border-accent-cyan/30 bg-accent-cyan/10 text-accent-cyan" : done ? "border-emerald/20 bg-emerald/10 text-emerald" : "border-border bg-surface text-text-muted")}>
                      {done && !active ? <Check className="h-4 w-4" aria-label="Complete" /> : <Icon className="h-4 w-4" />}
                    </span>
                    <div className="min-w-0"><p className={cn("text-sm font-medium",active ? "text-text-primary" : "text-text-secondary")}>{index+1}. {item.title}</p>
                      <p className="mt-0.5 hidden text-xs leading-relaxed text-text-muted lg:block">{item.description}</p></div>
                  </div>
                </li>;
              })}
            </ol>
          </nav>
          <p className="mt-7 hidden text-xs leading-relaxed text-text-muted lg:block">You can update your profile and address in Settings whenever your details change.</p>
        </aside>
        <section aria-labelledby="setup-title" className="overflow-hidden rounded-2xl border border-border bg-surface shadow-xl shadow-black/5 sm:rounded-3xl">
          <div className="space-y-6 p-5 sm:p-8 lg:p-10">
            <div className="border-b border-border pb-6">
              <p className="mb-2 text-xs font-medium text-accent-cyan">Step {currentIndex+1} of {flow.length}</p>
              <h1 id="setup-title" className="text-2xl font-semibold tracking-tight sm:text-3xl">{titles[step]}</h1>
              <p className="mt-2 text-sm leading-relaxed text-text-secondary">{step === "address" ? "Use your business address for a business account. Search to fill the fields, or enter them yourself." : step === "account" ? "Your registration details are already saved. Tell us what you’d like to do next." : step === "provider" ? "Review the details needed to offer GPUs and receive payments." : "Complete the checks required for your provider account."}</p>
            </div>
      {(step === "account" || step === "address") && <AccountProfileForm key={step} step={step}
        onSaved={async () => {
          if (step === "account") go("address");
          else if (providerIntent) go("provider");
          else router.replace(target);
        }} />}
      {step === "provider" && <ProviderProfileForm onSaved={async () => { await refreshUser({ throwOnError: true }); go("verification"); }} />}
      {step === "verification" && <div className="space-y-4">
        <p className="text-sm text-text-secondary">Stripe collects and checks the identity and payout information required for your provider account. If documents are needed, upload them directly in Stripe’s form.</p>
        {provider?.stripe_connected ? <StripeConnectEmbedded providerId={user.provider_id!} mode="setup" /> :
          <Button disabled={starting || !user.provider_id || !provider} onClick={async () => {
            setStarting(true); setError("");
            try {
              await registerProvider({provider_id:user.provider_id!,email:user.email,country:provider?.country,province:provider?.province,
                provider_type:user.account_type === "business" ? "company" : "individual", corporation_name:user.business_name});
              await refreshUser({ throwOnError: true }); await refreshProvider();
            } catch (err) { setError(err instanceof Error ? err.message : "Unable to start provider verification"); }
            finally { setStarting(false); }
          }}>{starting ? "Starting…" : "Start Stripe verification"}</Button>}
        {provider && !provider.stripe_connected && <p className="text-sm text-text-secondary">Your seller details are saved. Start Stripe verification to connect payouts.</p>}
        {provider?.stripe_connected && provider.payouts && <PayoutRequirements payouts={provider.payouts} />}
        {(provider?.stripe_connected || error) && <Button variant="outline" onClick={() => { void refreshProvider().catch((err) => setError(String(err))); }}>Refresh verification status</Button>}
        <p className="text-xs text-text-muted">Provider checks may take time. GPU hardware is verified separately when you register a host.</p>
        <Button onClick={() => router.replace(target)}>Continue to Xcelsior</Button>
      </div>}
      {error && <p role="alert" className="rounded-lg border border-red-400/25 bg-red-400/5 p-3 text-sm text-red-400">{error}</p>}
      {step !== "account" && <Button type="button" variant="outline" onClick={() => go(step === "address" ? "account" : step === "provider" ? "address" : "provider")}><ArrowLeft className="h-4 w-4" /> Back</Button>}
          </div>
          <footer className="border-t border-border bg-navy/30 px-5 py-4 sm:px-8 lg:px-10">
            <p className="break-all text-xs text-text-muted">Signed in as <span className="font-medium text-text-secondary">{user.email}</span></p>
          </footer>
        </section>
      </div>
    </div>
  </main>;

}

export default function OnboardingPage() {
  return <Suspense fallback={<p role="status" className="p-8">Loading account setup…</p>}><SetupFlow /></Suspense>;
}
