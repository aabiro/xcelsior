// messaging.ts — Honest, self-correcting status messages (Part F).
//
// PostHog shows "Finding and correcting errors" — honesty as polish. Instead of
// a generic "N checks failed" spinner, name the actual problem and signal that
// the wizard is guiding the fix. Pure & testable.

import type { CheckResult } from "./checks.js";

interface FailurePhrase {
    /** Regex matched against the failed check's name. */
    match: RegExp;
    /** Phrase naming the issue + the self-correcting intent. */
    phrase: string;
}

const FAILURE_PHRASES: FailurePhrase[] = [
    { match: /driver/i, phrase: "NVIDIA driver is too old — guiding you through the fix" },
    { match: /toolkit/i, phrase: "NVIDIA Container Toolkit needs attention — here's the fix" },
    { match: /runc/i, phrase: "runc is out of date — here's how to update it" },
    { match: /docker compose/i, phrase: "Docker Compose isn't available — let's set it up" },
    { match: /docker/i, phrase: "Docker isn't ready — let's get it running" },
    { match: /matmul|fp16|tflops|benchmark/i, phrase: "Benchmark didn't complete — let's retry" },
    { match: /pcie/i, phrase: "PCIe bandwidth is below target — re-measuring" },
    { match: /thermal|temp/i, phrase: "GPU is running hot — let it cool, then retry" },
    { match: /jitter|latency|packet|throughput|network/i, phrase: "Network quality is below target — checking your link" },
    { match: /wallet|balance/i, phrase: "Wallet balance is low — let's add funds" },
    { match: /admission/i, phrase: "Admission requirements aren't met yet — fixing versions" },
    { match: /verif|fingerprint|integrity|posture/i, phrase: "A verification check failed — guiding you through it" },
    { match: /api|connection|control plane/i, phrase: "Can't reach the control plane — retrying" },
    { match: /ssh/i, phrase: "SSH key setup needs attention — let's sort it out" },
    { match: /host registration|register/i, phrase: "Host registration didn't go through — retrying" },
];

/**
 * A concise, honest one-liner for a failed check set: names the primary issue
 * and conveys the wizard is helping. Falls back to the item name. Appends
 * "(+N more)" when several checks failed.
 */
export function summarizeFailure(items: CheckResult[]): string {
    const failed = items.filter((i) => !i.ok);
    if (failed.length === 0) return "All checks passed!";

    const primary = failed[0];
    const known = FAILURE_PHRASES.find((p) => p.match.test(primary.name));
    const base = known ? known.phrase : `${primary.name} needs attention — guiding you through the fix`;

    const extra = failed.length - 1;
    return extra > 0 ? `${base} (+${extra} more)` : base;
}
