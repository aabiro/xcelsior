// GPU display formatting — turns a canonical short title plus variant
// metadata into a user-facing string. Canonical short titles (from
// `./gpu-models`) NEVER contain VRAM, form factor, or "High frequency"
// markers; those live in dedicated columns and are composed into the
// display string here.

import type { FormFactor } from "./gpu-models";

export interface GpuDisplayParts {
    /** Canonical short title, e.g. "RTX 4090", "A100". */
    title: string;
    /** Optional VRAM capacity in GB. */
    vram_gb?: number | null;
    /** Optional form factor. "PCIe" is omitted from the output as default. */
    form_factor?: FormFactor | string | null;
    /** Whether the variant is paired with a high-frequency CPU. */
    high_frequency?: boolean | null;
}

/**
 * Compose a display string for a GPU SKU.
 *
 * Examples:
 *   formatGpuDisplay({ title: "RTX 4090", vram_gb: 24 })
 *     → "RTX 4090 24GB"
 *   formatGpuDisplay({ title: "A100", vram_gb: 80, form_factor: "SXM" })
 *     → "A100 80GB SXM"
 *   formatGpuDisplay({ title: "RTX 4090", vram_gb: 24, high_frequency: true })
 *     → "RTX 4090 24GB · High frequency"
 */
export function formatGpuDisplay(parts: GpuDisplayParts): string {
    const { title, vram_gb, form_factor, high_frequency } = parts;
    const pieces: string[] = [title];
    if (vram_gb && vram_gb > 0) pieces.push(`${vram_gb}GB`);
    if (form_factor && form_factor !== "PCIe") pieces.push(form_factor);
    const head = pieces.join(" ");
    return high_frequency ? `${head} · High frequency` : head;
}

/**
 * Short badge text for the high-frequency variant. Useful when the main
 * label needs to stay compact and the HF marker renders separately.
 */
export const HIGH_FREQ_BADGE = "HF";
