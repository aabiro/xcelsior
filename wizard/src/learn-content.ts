// learn-content.ts — Xcelsior-flavored Learn/Tips slides (Part C).
//
// PostHog teaches "events"; we teach the GPU marketplace: matching, XCU
// scoring, spot pricing, provider earnings. Charts are precomputed from
// ascii-viz at module load (those renderers are pure & deterministic), keeping
// the slide list plain data the slideshow just maps over.

import { sparkline, barChart, lineChart, pipeline, trimNumber } from "./ascii-viz.js";
import type { MarketplaceStats } from "./marketplace-stats.js";

export type SlideMode = "learn" | "tips";

export interface LearnSlide {
    /** Pane heading shown in brand color. */
    heading: string;
    /** Body paragraphs/lines (already wrapped short enough for a 40-col pane). */
    lines: string[];
    /** Optional precomputed chart lines, rendered under the body in gradient. */
    chart?: string[];
    /** Optional caption above the chart. */
    chartCaption?: string;
    mode: SlideMode;
}

// ── Sample data (illustrative, not live) ─────────────────────────────

const XCU_BARS = barChart([
    { label: "H100", value: 99 },
    { label: "A100", value: 78 },
    { label: "RTX 4090", value: 41 },
    { label: "RTX 3090", value: 28 },
], { width: 22, valueMode: "value" });

const SPOT_SPARK = sparkline([42, 38, 45, 40, 33, 36, 30, 28, 34, 31, 26, 29]);

const EARNINGS_LINE = lineChart([120, 180, 240, 210, 320, 380, 460, 540], {
    width: 26, height: 5, axis: true,
});

const AVAILABILITY = barChart([
    { label: "Available", value: 64 },
    { label: "In use", value: 36 },
], { width: 22 });

const JOB_PIPELINE = pipeline(["submit", "schedule", "run", "settle"], 2);

const UTIL_BARS = barChart([
    { label: "GPU 0", value: 92 },
    { label: "GPU 1", value: 74 },
    { label: "GPU 2", value: 18 },
], { width: 22 });

// ── Slides ───────────────────────────────────────────────────────────

export const LEARN_SLIDES: LearnSlide[] = [
    {
        mode: "learn",
        heading: "How the marketplace works",
        lines: [
            "Xcelsior matches your job to the best",
            "available GPU across many providers —",
            "by VRAM, price, region and reputation.",
        ],
        chartCaption: "Job lifecycle",
        chart: [JOB_PIPELINE],
    },
    {
        mode: "learn",
        heading: "XCU — one score for compute",
        lines: [
            "Every GPU is benchmarked (FP16 matmul,",
            "PCIe, thermals) into an XCU score so you",
            "compare apples to apples, not model names.",
        ],
        chartCaption: "XCU score by GPU",
        chart: XCU_BARS,
    },
    {
        mode: "learn",
        heading: "Spot & interruptible pricing",
        lines: [
            "Spot instances run on idle capacity at a",
            "fraction of on-demand price. Set a floor;",
            "the market clears the rest.",
        ],
        chartCaption: "Spot ¢/hr (last 12 ticks)",
        chart: [SPOT_SPARK],
    },
    {
        mode: "learn",
        heading: "Providers earn while idle",
        lines: [
            "Share a GPU and earn per second of use.",
            "Reputation tiers unlock higher rates and",
            "priority placement over time.",
        ],
        chartCaption: "Provider earnings (CAD/wk)",
        chart: EARNINGS_LINE,
    },
    {
        mode: "learn",
        heading: "Aim for ~40 hours a week",
        lines: [
            "No need to run 24/7. ~40 online hours a",
            "week is the sweet spot — evenings and",
            "overnight clears it. Predictable wins.",
        ],
    },
    {
        mode: "learn",
        heading: "Don't vanish mid-job",
        lines: [
            "Staying online lifts your reliability —",
            "and reliability multiplies your whole",
            "score. Drop a job and you lose a tier.",
        ],
    },
    {
        mode: "learn",
        heading: "Live marketplace depth",
        lines: [
            "Capacity ebbs and flows. The scheduler",
            "places your job the moment a matching GPU",
            "frees up — no manual hunting.",
        ],
        chartCaption: "Fleet availability",
        chart: AVAILABILITY,
    },
    {
        mode: "learn",
        heading: "Everything Xcelsior does",
        lines: [
            "◆ Marketplace      ◆ Spot instances",
            "◆ Serverless infer ◆ Persistent volumes",
            "◆ Hardware verify  ◆ Earnings dashboard",
        ],
    },
    {
        mode: "tips",
        heading: "Tip — utilization at a glance",
        lines: [
            "Track per-GPU utilization from the",
            "dashboard to spot idle capacity worth",
            "renting out.",
        ],
        chartCaption: "GPU utilization",
        chart: UTIL_BARS,
    },
    {
        mode: "tips",
        heading: "Tip — ask Hexara anything",
        lines: [
            "Press ? at almost any step to ask Hexara",
            "for help — she can read your check results",
            "and walk you through fixes.",
        ],
    },
];

/**
 * Step-aware time expectations — shown above the cycling card so a long wait
 * states what's happening and roughly how long ("here's X while you wait").
 * Keyed by step id; returns null for steps without a meaningful wait.
 */
const STEP_TIME_HINTS: Record<string, string> = {
    "benchmark": "Benchmarks take ~60s — here's how XCU scoring works while you wait ☕",
    "verification": "7-point hardware verification — just a few seconds.",
    "network-bench": "Measuring latency, jitter & throughput to the scheduler…",
    "network-setup": "Setting up secure mesh networking…",
    "host-register": "Registering your host on the marketplace…",
    "admission-gate": "Checking admission & the security runtime…",
    "launch-instance": "Spinning up your instance — this can take a few minutes.",
    "browse-gpus": "Searching the marketplace for the best match…",
    "gpu-detect": "Detecting local GPUs via nvidia-smi…",
    "version-check": "Checking runc, Docker, NVIDIA driver & toolkit versions…",
    "docker-check": "Probing your Docker environment…",
    "wallet-check": "Checking your wallet balance…",
    "ssh-key-setup": "Setting up SSH keys for instance access…",
    "api-check": "Verifying the control-plane connection…",
    "sdk-detect": "Scanning for package.json / pyproject.toml in your project tree…",
    "sdk-install": "Checking node_modules for @xcelsior-gpu/sdk…",
    "sdk-credentials": "Writing .env.local and provisioning OAuth for automation…",
    "sdk-verify": "Calling the API with your xoa_ token…",
};

export function timeHintForStep(stepId: string): string | null {
    return STEP_TIME_HINTS[stepId] ?? null;
}

/**
 * Build Learn slides from a LIVE marketplace snapshot (Part C — data-driven,
 * not static art). Returns [] if there's nothing meaningful to show, letting
 * the caller fall back to the concept cards.
 */
export function buildMarketplaceSlides(stats: MarketplaceStats): LearnSlide[] {
    if (!stats || stats.count === 0) return [];
    const slides: LearnSlide[] = [];
    const top = stats.byModel.slice(0, 4);

    if (top.length > 0) {
        slides.push({
            mode: "learn",
            heading: "Live marketplace — price by GPU",
            lines: [
                `${stats.count} GPUs listed · ${stats.activeCount} available now.`,
                `Avg $${trimNumber(stats.avgPrice)}/hr across the fleet.`,
            ],
            chartCaption: "Avg $/hr by model (live)",
            chart: barChart(
                top.map((m) => ({ label: m.model, value: m.avgPrice })),
                { width: 20, valueMode: "value" },
            ),
        });
    }

    if (stats.prices.length >= 3) {
        slides.push({
            mode: "learn",
            heading: "Live price range",
            lines: [
                `Low $${trimNumber(stats.minPrice)}/hr · high $${trimNumber(stats.maxPrice)}/hr.`,
                "Pick by budget or performance — the scheduler matches you.",
            ],
            chartCaption: "Listing prices, low → high (live)",
            chart: [sparkline(stats.prices)],
        });
    }

    const byDemand = [...stats.byModel].filter((m) => m.totalJobs > 0).sort((a, b) => b.totalJobs - a.totalJobs).slice(0, 4);
    if (byDemand.length > 0) {
        slides.push({
            mode: "learn",
            heading: "Most in-demand right now",
            lines: ["Jobs run per GPU model — where the demand is."],
            chartCaption: "Total jobs by model (live)",
            chart: barChart(
                byDemand.map((m) => ({ label: m.model, value: m.totalJobs })),
                { width: 20, valueMode: "value" },
            ),
        });
    }

    return slides;
}

/** Pure helper: index of the next slide, wrapping. Exported for tests. */
export function nextSlideIndex(current: number, total: number): number {
    if (total <= 0) return 0;
    return (current + 1) % total;
}

const SDK_FLOW = pipeline(["Your App", "SDK", "API", "Dashboard"], 1);

/** SDK integration track — mirrors PostHog's "Here's the flow" diagram. */
export const SDK_LEARN_SLIDES: LearnSlide[] = [
    {
        mode: "learn",
        heading: "Here's the flow",
        lines: [
            "Your app calls the Xcelsior SDK with",
            "your xoa_ token. The SDK talks to",
            "our API — you query GPUs, launch jobs,",
            "and manage billing from code.",
        ],
        chartCaption: "Data path",
        chart: [SDK_FLOW],
    },
    {
        mode: "learn",
        heading: "Session tokens (xoa_)",
        lines: [
            "Device sign-in gives you an xoa_",
            "access token for your user session.",
            "OAuth client IDs (oauth_…) are for",
            "server-to-server — not the same thing.",
        ],
    },
    {
        mode: "tips",
        heading: "OAuth for automation",
        lines: [
            "The wizard creates a confidential",
            "OAuth client in Settings → API.",
            "Use client_credentials for CI and",
            "long-running workers — no browser.",
        ],
    },
    {
        mode: "tips",
        heading: "TypeScript-first",
        lines: [
            "npm install @xcelsior-gpu/sdk",
            "Full types for instances, marketplace,",
            "billing, and serverless endpoints.",
        ],
    },
];

/** Slides filtered to a mode (learn slides always show; tips are extra nudges). */
export function slidesForMode(mode: SlideMode): LearnSlide[] {
    if (mode === "tips") return LEARN_SLIDES;
    return LEARN_SLIDES.filter((s) => s.mode === "learn");
}

export function sdkLearnSlides(): LearnSlide[] {
    return SDK_LEARN_SLIDES;
}
