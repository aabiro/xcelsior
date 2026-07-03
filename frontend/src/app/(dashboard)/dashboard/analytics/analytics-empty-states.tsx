"use client";

import Link from "next/link";
import { BarChart3, Clock, Cpu, DollarSign, Globe, Server, Shield, TrendingUp, Wallet, Zap } from "lucide-react";
import { Card } from "@/components/ui/card";
import { StatCard } from "@/components/ui/stat-card";
import { FadeIn } from "@/components/ui/motion";
import { TopGpuChart } from "./charts";

export function AnalyticsTabEmpty({
  icon: Icon,
  title,
  description,
  steps,
}: {
  icon: React.ElementType;
  title: string;
  description: string;
  steps: { n: number; title: string; body: React.ReactNode }[];
}) {
  return (
    <FadeIn>
      <div className="flex flex-col items-center justify-center py-16">
        <div className="flex h-20 w-20 items-center justify-center rounded-2xl bg-surface mb-6 ring-1 ring-border/60">
          <Icon className="h-10 w-10 text-text-muted" />
        </div>
        <h3 className="text-xl font-semibold mb-2">{title}</h3>
        <p className="text-sm text-text-secondary text-center max-w-md mb-6">{description}</p>
        <div className="rounded-xl border border-border/30 bg-surface/60 px-5 py-4 max-w-lg w-full text-left">
          <p className="text-xs font-semibold text-text-secondary uppercase tracking-wide mb-3">How to unlock</p>
          <div className="space-y-2.5">
            {steps.map((s) => (
              <div key={s.n} className="flex items-start gap-2.5">
                <div className="mt-0.5 flex h-4.5 w-4.5 shrink-0 items-center justify-center rounded-full bg-accent-cyan/15 text-accent-cyan text-[9px] font-bold">
                  {s.n}
                </div>
                <p className="text-xs text-text-secondary">
                  <span className="font-medium text-text-primary">{s.title}</span>, {s.body}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </FadeIn>
  );
}

export function PlatformPulseOverview({
  spotPrices,
  marketplaceStats,
  walletBalance,
  leaderboardCount,
  gpuModelsAvailable,
}: {
  spotPrices: { model: string; price: number }[];
  marketplaceStats: { total_offers: number; total_gpus: number; avg_price: number } | null;
  walletBalance: number | null;
  leaderboardCount: number;
  gpuModelsAvailable: number;
}) {
  const spotChart = spotPrices.slice(0, 8).map((s) => ({
    name: s.model,
    spend: s.price,
    jobs: 0,
    hours: 0,
  }));

  return (
    <div className="space-y-6">
      <div className="rounded-2xl border border-accent-cyan/15 bg-gradient-to-br from-accent-cyan/5 via-surface to-accent-violet/5 p-5 md:p-6">
        <div className="flex items-center gap-2 mb-1">
          <Globe className="h-4 w-4 text-accent-cyan" />
          <h3 className="text-sm font-semibold">Platform pulse</h3>
        </div>
        <p className="text-xs text-text-secondary max-w-2xl">
          Live marketplace and wallet data you can see right away, personal job analytics appear after your first instance or hosted job.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <StatCard
          label="GPUs listed"
          value={marketplaceStats?.total_gpus ?? gpuModelsAvailable ?? "-"}
          icon={Server}
          glow="cyan"
        />
        <StatCard
          label="Market offers"
          value={marketplaceStats?.total_offers ?? "-"}
          icon={TrendingUp}
          glow="emerald"
        />
        <StatCard
          label="Wallet balance"
          value={walletBalance != null ? `$${walletBalance.toFixed(2)}` : "-"}
          icon={Wallet}
          glow="gold"
        />
        <StatCard
          label="Top providers"
          value={leaderboardCount > 0 ? leaderboardCount : "-"}
          icon={Shield}
          glow="violet"
        />
      </div>

      {spotChart.length > 0 && (
        <FadeIn delay={0.1}>
          <div>
            <div className="flex items-center gap-2 mb-3">
              <Zap className="h-4 w-4 text-accent-gold" />
              <h3 className="text-sm font-semibold">Current spot rates</h3>
            </div>
            <TopGpuChart data={spotChart} />
          </div>
        </FadeIn>
      )}

      <div className="flex flex-col items-center py-8">
        <div className="relative flex h-20 w-20 items-center justify-center rounded-2xl bg-surface mb-5 ring-1 ring-border/50">
          <BarChart3 className="h-10 w-10 text-text-muted" />
          <div className="absolute -top-1 -right-1 h-3.5 w-3.5 rounded-full bg-accent-cyan/40 animate-pulse" />
        </div>
        <h3 className="text-lg font-semibold mb-2">Your usage charts are waiting</h3>
        <p className="text-sm text-text-secondary max-w-md text-center mb-6">
          Launch an instance or register a host to unlock spend trends, GPU hours, and auto-insights tailored to your account.
        </p>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3 w-full max-w-lg">
          <Card className="p-4 text-center border-dashed hover:border-accent-cyan/30 transition-colors">
            <TrendingUp className="h-5 w-5 text-accent-cyan mx-auto mb-2" />
            <p className="text-xs text-text-muted">Spend trends</p>
            <p className="text-base font-bold font-mono text-text-muted/40 mt-1">-</p>
          </Card>
          <Card className="p-4 text-center border-dashed hover:border-emerald/30 transition-colors">
            <Cpu className="h-5 w-5 text-emerald mx-auto mb-2" />
            <p className="text-xs text-text-muted">GPU hours</p>
            <p className="text-base font-bold font-mono text-text-muted/40 mt-1">-</p>
          </Card>
          <Card className="p-4 text-center border-dashed hover:border-accent-gold/30 transition-colors">
            <Clock className="h-5 w-5 text-accent-gold mx-auto mb-2" />
            <p className="text-xs text-text-muted">Job insights</p>
            <p className="text-base font-bold font-mono text-text-muted/40 mt-1">-</p>
          </Card>
        </div>
        <div className="mt-6 flex flex-wrap justify-center gap-3">
          <Link href="/dashboard/instances" className="text-sm text-accent-cyan hover:underline font-medium">
            Launch an instance →
          </Link>
          <Link href="/dashboard/hosts" className="text-sm text-accent-violet hover:underline font-medium">
            Register a host →
          </Link>
        </div>
      </div>
    </div>
  );
}

export function ComputeTabEmpty() {
  return (
    <AnalyticsTabEmpty
      icon={Cpu}
      title="No compute activity yet"
      description="GPU utilization, jurisdiction splits, and performance breakdowns appear once you run workloads."
      steps={[
        {
          n: 1,
          title: "Launch a GPU instance",
          body: (
            <>
              Start from{" "}
              <Link href="/dashboard/instances" className="text-accent-cyan hover:underline">
                Instances
              </Link>{" "}
             , even a short test job unlocks utilization charts.
            </>
          ),
        },
        {
          n: 2,
          title: "Try different GPU models",
          body: "Compare performance radar and duration histograms across models you use.",
        },
      ]}
    />
  );
}

export function FinancialTabEmpty() {
  return (
    <AnalyticsTabEmpty
      icon={DollarSign}
      title="No spend data yet"
      description="Wallet activity, cumulative spend, and cost-per-hour trends need at least one billed job or deposit."
      steps={[
        {
          n: 1,
          title: "Add wallet credits",
          body: (
            <>
              Visit{" "}
              <Link href="/dashboard/billing" className="text-accent-cyan hover:underline">
                Billing
              </Link>{" "}
              to fund your account, balance shows here immediately.
            </>
          ),
        },
        {
          n: 2,
          title: "Run a billed instance",
          body: "Spend trends and cost efficiency charts populate automatically.",
        },
      ]}
    />
  );
}