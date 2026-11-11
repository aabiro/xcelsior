"use client";

import { Suspense, useEffect } from "react";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { useAuth } from "@/lib/auth";
import { useTheme } from "@/lib/theme";
import { normalizeAuthRedirectPath } from "@/lib/api";
import { AccountProfileForm } from "@/components/ui/province-selector";
import { ProviderProfileForm } from "@/components/providers/seller-profile-card";
import { Button } from "@/components/ui/button";
import { SITE_ASSETS } from "@/lib/brand-assets";
import { cn } from "@/lib/utils";
import { ArrowLeft, Check, MapPin, Moon, Sun, UserRound } from "lucide-react";

function SetupFlow() {
  const { user, loading, refreshUser } = useAuth();
  const { theme, toggleTheme } = useTheme();
  const router = useRouter(); const params = useParams<{ step?: string[] }>(); const search = useSearchParams();
  const requested = params.step?.[0] || "account";
  const payout = requested === "provider";
  const step = requested === "address" ? "address" : "account";
  const rawTarget = normalizeAuthRedirectPath(search.get("redirect"), "/dashboard");
  const target = rawTarget.startsWith("/onboarding") ? "/dashboard" : rawTarget;
  const resumePayout = payout || search.get("resume") === "provider";
  const resume = resumePayout ? "&resume=provider" : "";
  const go = (next: string) => router.push(`/onboarding/${next}?redirect=${encodeURIComponent(target)}${resume}`);
  useEffect(() => {
    if (requested === "verification") router.replace("/dashboard/earnings");
  }, [requested, router]);
  useEffect(() => {
    if (!loading && !user) router.replace(`/login?redirect=${encodeURIComponent(`/onboarding/${requested}?redirect=${encodeURIComponent(target)}`)}`);
  }, [loading, user, router, requested, target]);
  useEffect(() => {
    if (!user || requested === "verification" || requested === "account" || !requested) return;
    const missing = user.missing_profile_fields || [];
    const accountMissing = ["name", "email", "account_type", "business_name"].some((field) => missing.includes(field));
    const addressMissing = ["country", "province", "address_line1", "city", "postal_code"].some((field) => missing.includes(field));
    const next = accountMissing ? "account" : requested !== "address" && addressMissing ? "address" : null;
    if (next) router.replace(`/onboarding/${next}?redirect=${encodeURIComponent(target)}${resumePayout ? "&resume=provider" : ""}`);
  }, [user, requested, resumePayout, router, target]);
  if (loading || !user || requested === "verification") return <p role="status" className="p-8">Loading your account…</p>;
  const titles: Record<string, string> = { account: "Your account", address: "Your address", provider: "Payout details" };
  const flow = [
    { id: "account", title: "Your account", description: "Personal or business", icon: UserRound },
    { id: "address", title: "Your address", description: "Accurate details, entered once", icon: MapPin },
  ];
  const currentIndex = Math.max(flow.findIndex((item) => item.id === step), 0);
  const missing = user.missing_profile_fields || [];
  const accountComplete = user.profile_complete === true || !["name", "email", "account_type", "business_name"].some((field) => missing.includes(field));
  const addressComplete = user.profile_complete === true || !["country", "province", "address_line1", "city", "postal_code"].some((field) => missing.includes(field));
  const subtitle = payout
    ? "Legal name and tax identifier, whenever you want payouts. Stripe setup stays on Earnings."
    : step === "address"
      ? "Search for your address, or enter it yourself. You can update it anytime in Settings."
      : "Choose a personal or business account. Payout setup waits until you provide a GPU.";
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
          <p className="mt-3 max-w-sm text-sm leading-relaxed text-text-secondary">A few details to get you started.</p>
          {!payout && <nav aria-label="Setup progress" className="mt-7 lg:mt-10">
            <ol className="grid grid-cols-2 gap-2 lg:grid-cols-1 lg:gap-3">
              {flow.map((item, index) => {
                const active = step === item.id;
                const done = item.id === "account" ? accountComplete : addressComplete;
                const canNavigate = !active && (item.id === "account" || accountComplete);
                const Icon = item.icon;
                const content = (
                  <div className={cn("flex items-center gap-3 rounded-xl border p-3 text-left transition-colors", active ? "border-accent-cyan/30 bg-accent-cyan/5" : canNavigate ? "border-transparent hover:bg-surface-hover" : "border-transparent")}>
                    <span className={cn("flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border", active ? "border-accent-cyan/30 bg-accent-cyan/10 text-accent-cyan" : done ? "border-emerald/20 bg-emerald/10 text-emerald" : "border-border bg-surface text-text-muted")}>
                      {done && !active ? <Check className="h-4 w-4" aria-label="Complete" /> : <Icon className="h-4 w-4" />}
                    </span>
                    <div className="min-w-0"><p className={cn("text-sm font-medium", active ? "text-text-primary" : "text-text-secondary")}>{index + 1}. {item.title}</p>
                      <p className="mt-0.5 hidden text-xs leading-relaxed text-text-muted lg:block">{item.description}</p></div>
                  </div>
                );
                return <li key={item.id} aria-current={active ? "step" : undefined}>
                  {canNavigate ? <button type="button" onClick={() => go(item.id)} className="w-full rounded-xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ice-blue">{content}</button> : content}
                </li>;
              })}
            </ol>
          </nav>}
          <p className="mt-7 hidden text-xs leading-relaxed text-text-muted lg:block">You can update your address in Settings. Stripe payout setup stays on Earnings until you provide a GPU.</p>
        </aside>
        <section aria-labelledby="setup-title" className="overflow-hidden rounded-2xl border border-border bg-surface shadow-xl shadow-black/5 sm:rounded-3xl">
          <div className="space-y-6 p-5 sm:p-8 lg:p-10">
            <div className="border-b border-border pb-6">
              {!payout && <p className="mb-2 text-xs font-medium text-accent-cyan">Step {currentIndex + 1} of {flow.length}</p>}
              <h1 id="setup-title" className="text-2xl font-semibold tracking-tight sm:text-3xl">{titles[payout ? "provider" : step]}</h1>
              <p className="mt-2 text-sm leading-relaxed text-text-secondary">{subtitle}</p>
            </div>
            {(step === "account" || step === "address") && !payout && <AccountProfileForm key={step} step={step}
              onSaved={async () => {
                if (step === "account") go("address");
                else if (resumePayout) router.replace(`/onboarding/provider?redirect=${encodeURIComponent(target)}`);
                else router.replace(target);
              }} />}
            {payout && <ProviderProfileForm mode="onboarding" onEditAddress={() => go("address")} onSaved={async () => { await refreshUser({ throwOnError: true }); router.replace(target); }} />}
            {step === "address" && !payout && <Button type="button" variant="outline" onClick={() => go("account")}><ArrowLeft className="h-4 w-4" /> Back</Button>}
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
