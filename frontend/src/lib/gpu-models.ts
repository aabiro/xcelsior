// GPU model catalog — canonical short titles that must match:
//   - worker_agent.py _NVIDIA_SMI_NAME_MAP values
//   - db.py _GPU_PRICING_BASE first column
//
// Titles are short-form ONLY (e.g. "RTX 4090", "A100"). Variants such
// as VRAM capacity, form factor (PCIe/SXM/OAM), and high-frequency CPU
// pairings are expressed as SEPARATE FIELDS and must never be encoded
// into the title. Use `formatGpuDisplay()` from `./gpu-display` to
// render a user-facing string that includes variant details.

export type FormFactor = "PCIe" | "SXM" | "OAM";

export interface GpuModel {
    /** Canonical short title — stable identifier AND display label. */
    value: string;
    /** Same as `value`. Kept for legacy callers. */
    label: string;
    /** Primary (most common) VRAM capacity. */
    vram_gb: number;
    /** Additional VRAM capacities this SKU ships in (excludes primary). */
    vram_options?: number[];
    /** Form factors this SKU is available in. Defaults to ["PCIe"]. */
    form_factors?: FormFactor[];
    /** True if a high-frequency CPU pairing is offered. */
    supports_high_freq?: boolean;
    category: GpuCategory;
}

export type GpuCategory =
    | "NVIDIA Data Center"
    | "NVIDIA RTX 50 Series"
    | "NVIDIA RTX 40 Series"
    | "NVIDIA RTX 30 Series"
    | "NVIDIA RTX 20 Series"
    | "NVIDIA Workstation"
    | "AMD Data Center"
    | "AMD Consumer";

export const GPU_CATEGORIES: GpuCategory[] = [
    "NVIDIA Data Center",
    "NVIDIA RTX 50 Series",
    "NVIDIA RTX 40 Series",
    "NVIDIA RTX 30 Series",
    "NVIDIA RTX 20 Series",
    "NVIDIA Workstation",
    "AMD Data Center",
    "AMD Consumer",
];

export const GPU_MODELS: GpuModel[] = [
    // ── NVIDIA Data Center ──
    { value: "H200", label: "H200", vram_gb: 141, form_factors: ["SXM"], category: "NVIDIA Data Center" },
    { value: "H100", label: "H100", vram_gb: 80, form_factors: ["SXM", "PCIe"], category: "NVIDIA Data Center" },
    { value: "H100 NVL", label: "H100 NVL", vram_gb: 94, form_factors: ["PCIe"], category: "NVIDIA Data Center" },
    { value: "A100", label: "A100", vram_gb: 80, vram_options: [40], form_factors: ["SXM", "PCIe"], category: "NVIDIA Data Center" },
    { value: "A40", label: "A40", vram_gb: 48, form_factors: ["PCIe"], category: "NVIDIA Data Center" },
    { value: "A30", label: "A30", vram_gb: 24, form_factors: ["PCIe"], category: "NVIDIA Data Center" },
    { value: "A10", label: "A10", vram_gb: 24, form_factors: ["PCIe"], category: "NVIDIA Data Center" },
    { value: "A16", label: "A16", vram_gb: 16, form_factors: ["PCIe"], category: "NVIDIA Data Center" },
    { value: "L40S", label: "L40S", vram_gb: 48, form_factors: ["PCIe"], category: "NVIDIA Data Center" },
    { value: "L40", label: "L40", vram_gb: 48, form_factors: ["PCIe"], category: "NVIDIA Data Center" },
    { value: "L4", label: "L4", vram_gb: 24, form_factors: ["PCIe"], category: "NVIDIA Data Center" },
    { value: "T4", label: "T4", vram_gb: 16, form_factors: ["PCIe"], category: "NVIDIA Data Center" },
    { value: "V100", label: "V100", vram_gb: 32, vram_options: [16], form_factors: ["SXM", "PCIe"], category: "NVIDIA Data Center" },

    // ── NVIDIA RTX 50 Series ──
    { value: "RTX 5090", label: "RTX 5090", vram_gb: 32, category: "NVIDIA RTX 50 Series" },
    { value: "RTX 5080", label: "RTX 5080", vram_gb: 16, category: "NVIDIA RTX 50 Series" },
    { value: "RTX 5070 Ti", label: "RTX 5070 Ti", vram_gb: 16, category: "NVIDIA RTX 50 Series" },
    { value: "RTX 5070", label: "RTX 5070", vram_gb: 12, category: "NVIDIA RTX 50 Series" },
    { value: "RTX 5060 Ti", label: "RTX 5060 Ti", vram_gb: 16, category: "NVIDIA RTX 50 Series" },
    { value: "RTX 5060", label: "RTX 5060", vram_gb: 8, category: "NVIDIA RTX 50 Series" },

    // ── NVIDIA RTX 40 Series ──
    { value: "RTX 4090", label: "RTX 4090", vram_gb: 24, supports_high_freq: true, category: "NVIDIA RTX 40 Series" },
    { value: "RTX 4080 Super", label: "RTX 4080 Super", vram_gb: 16, category: "NVIDIA RTX 40 Series" },
    { value: "RTX 4080", label: "RTX 4080", vram_gb: 16, category: "NVIDIA RTX 40 Series" },
    { value: "RTX 4070 Ti Super", label: "RTX 4070 Ti Super", vram_gb: 16, category: "NVIDIA RTX 40 Series" },
    { value: "RTX 4070 Ti", label: "RTX 4070 Ti", vram_gb: 12, category: "NVIDIA RTX 40 Series" },
    { value: "RTX 4070 Super", label: "RTX 4070 Super", vram_gb: 12, category: "NVIDIA RTX 40 Series" },
    { value: "RTX 4070", label: "RTX 4070", vram_gb: 12, category: "NVIDIA RTX 40 Series" },
    { value: "RTX 4060 Ti", label: "RTX 4060 Ti", vram_gb: 16, vram_options: [8], category: "NVIDIA RTX 40 Series" },
    { value: "RTX 4060", label: "RTX 4060", vram_gb: 8, category: "NVIDIA RTX 40 Series" },

    // ── NVIDIA RTX 30 Series ──
    { value: "RTX 3090 Ti", label: "RTX 3090 Ti", vram_gb: 24, category: "NVIDIA RTX 30 Series" },
    { value: "RTX 3090", label: "RTX 3090", vram_gb: 24, category: "NVIDIA RTX 30 Series" },
    { value: "RTX 3080 Ti", label: "RTX 3080 Ti", vram_gb: 12, category: "NVIDIA RTX 30 Series" },
    { value: "RTX 3080", label: "RTX 3080", vram_gb: 12, vram_options: [10], category: "NVIDIA RTX 30 Series" },
    { value: "RTX 3070 Ti", label: "RTX 3070 Ti", vram_gb: 8, category: "NVIDIA RTX 30 Series" },
    { value: "RTX 3070", label: "RTX 3070", vram_gb: 8, category: "NVIDIA RTX 30 Series" },
    { value: "RTX 3060 Ti", label: "RTX 3060 Ti", vram_gb: 8, category: "NVIDIA RTX 30 Series" },
    { value: "RTX 3060", label: "RTX 3060", vram_gb: 12, category: "NVIDIA RTX 30 Series" },

    // ── NVIDIA RTX 20 Series ──
    { value: "RTX 2080 Ti", label: "RTX 2080 Ti", vram_gb: 11, category: "NVIDIA RTX 20 Series" },
    { value: "RTX 2080 Super", label: "RTX 2080 Super", vram_gb: 8, category: "NVIDIA RTX 20 Series" },
    { value: "RTX 2080", label: "RTX 2080", vram_gb: 8, category: "NVIDIA RTX 20 Series" },
    { value: "RTX 2070 Super", label: "RTX 2070 Super", vram_gb: 8, category: "NVIDIA RTX 20 Series" },
    { value: "RTX 2070", label: "RTX 2070", vram_gb: 8, category: "NVIDIA RTX 20 Series" },
    { value: "RTX 2060 Super", label: "RTX 2060 Super", vram_gb: 8, category: "NVIDIA RTX 20 Series" },
    { value: "RTX 2060", label: "RTX 2060", vram_gb: 6, category: "NVIDIA RTX 20 Series" },

    // ── NVIDIA Workstation ──
    { value: "RTX 6000 Ada", label: "RTX 6000 Ada", vram_gb: 48, category: "NVIDIA Workstation" },
    { value: "RTX 5000 Ada", label: "RTX 5000 Ada", vram_gb: 32, category: "NVIDIA Workstation" },
    { value: "RTX 4000 Ada", label: "RTX 4000 Ada", vram_gb: 20, category: "NVIDIA Workstation" },
    { value: "RTX A6000", label: "RTX A6000", vram_gb: 48, category: "NVIDIA Workstation" },
    { value: "RTX A5000", label: "RTX A5000", vram_gb: 24, category: "NVIDIA Workstation" },
    { value: "RTX A4000", label: "RTX A4000", vram_gb: 16, category: "NVIDIA Workstation" },

    // ── AMD Data Center ──
    { value: "MI300X", label: "MI300X", vram_gb: 192, form_factors: ["OAM"], category: "AMD Data Center" },
    { value: "MI250X", label: "MI250X", vram_gb: 128, form_factors: ["OAM"], category: "AMD Data Center" },
    { value: "MI210", label: "MI210", vram_gb: 64, form_factors: ["PCIe"], category: "AMD Data Center" },

    // ── AMD Consumer ──
    { value: "RX 7900 XTX", label: "RX 7900 XTX", vram_gb: 24, category: "AMD Consumer" },
    { value: "RX 7900 XT", label: "RX 7900 XT", vram_gb: 20, category: "AMD Consumer" },
];

export function getGpuModel(value: string): GpuModel | null {
    return GPU_MODELS.find((g) => g.value === value) ?? null;
}

export function getVramForModel(model: string): number | null {
    const entry = getGpuModel(model);
    return entry ? entry.vram_gb : null;
}

/** All VRAM capacities a model ships in, ascending. */
export function getVramOptions(model: string): number[] {
    const entry = getGpuModel(model);
    if (!entry) return [];
    const all = [entry.vram_gb, ...(entry.vram_options ?? [])];
    return [...new Set(all)].sort((a, b) => a - b);
}

export function getGpusByCategory(): Record<GpuCategory, GpuModel[]> {
    const grouped = {} as Record<GpuCategory, GpuModel[]>;
    for (const cat of GPU_CATEGORIES) {
        grouped[cat] = GPU_MODELS.filter((g) => g.category === cat);
    }
    return grouped;
}
