export interface GpuModel {
    value: string;
    label: string;
    vram_gb: number;
    category: GpuCategory;
}

export type GpuCategory =
    | "NVIDIA Data Center"
    | "NVIDIA RTX 50 Series"
    | "NVIDIA RTX 40 Series"
    | "NVIDIA RTX 30 Series"
    | "NVIDIA Workstation"
    | "AMD Data Center"
    | "AMD Consumer";

export const GPU_CATEGORIES: GpuCategory[] = [
    "NVIDIA Data Center",
    "NVIDIA RTX 50 Series",
    "NVIDIA RTX 40 Series",
    "NVIDIA RTX 30 Series",
    "NVIDIA Workstation",
    "AMD Data Center",
    "AMD Consumer",
];

export const GPU_MODELS: GpuModel[] = [
    // ── NVIDIA Data Center ──
    { value: "H200", label: "NVIDIA H200", vram_gb: 141, category: "NVIDIA Data Center" },
    { value: "H100-80GB", label: "NVIDIA H100 80GB", vram_gb: 80, category: "NVIDIA Data Center" },
    { value: "H100-NVL", label: "NVIDIA H100 NVL", vram_gb: 94, category: "NVIDIA Data Center" },
    { value: "A100-80GB", label: "NVIDIA A100 80GB", vram_gb: 80, category: "NVIDIA Data Center" },
    { value: "A100-40GB", label: "NVIDIA A100 40GB", vram_gb: 40, category: "NVIDIA Data Center" },
    { value: "A40", label: "NVIDIA A40", vram_gb: 48, category: "NVIDIA Data Center" },
    { value: "A30", label: "NVIDIA A30", vram_gb: 24, category: "NVIDIA Data Center" },
    { value: "A10", label: "NVIDIA A10", vram_gb: 24, category: "NVIDIA Data Center" },
    { value: "A16", label: "NVIDIA A16", vram_gb: 16, category: "NVIDIA Data Center" },
    { value: "L40S", label: "NVIDIA L40S", vram_gb: 48, category: "NVIDIA Data Center" },
    { value: "L40", label: "NVIDIA L40", vram_gb: 48, category: "NVIDIA Data Center" },
    { value: "L4", label: "NVIDIA L4", vram_gb: 24, category: "NVIDIA Data Center" },
    { value: "T4", label: "NVIDIA T4", vram_gb: 16, category: "NVIDIA Data Center" },
    { value: "V100-32GB", label: "NVIDIA V100 32GB", vram_gb: 32, category: "NVIDIA Data Center" },
    { value: "V100-16GB", label: "NVIDIA V100 16GB", vram_gb: 16, category: "NVIDIA Data Center" },

    // ── NVIDIA RTX 50 Series ──
    { value: "RTX-5090", label: "NVIDIA GeForce RTX 5090", vram_gb: 32, category: "NVIDIA RTX 50 Series" },
    { value: "RTX-5080", label: "NVIDIA GeForce RTX 5080", vram_gb: 16, category: "NVIDIA RTX 50 Series" },
    { value: "RTX-5070-Ti", label: "NVIDIA GeForce RTX 5070 Ti", vram_gb: 16, category: "NVIDIA RTX 50 Series" },
    { value: "RTX-5070", label: "NVIDIA GeForce RTX 5070", vram_gb: 12, category: "NVIDIA RTX 50 Series" },
    { value: "RTX-5060-Ti", label: "NVIDIA GeForce RTX 5060 Ti", vram_gb: 16, category: "NVIDIA RTX 50 Series" },
    { value: "RTX-5060", label: "NVIDIA GeForce RTX 5060", vram_gb: 8, category: "NVIDIA RTX 50 Series" },

    // ── NVIDIA RTX 40 Series ──
    { value: "RTX-4090", label: "NVIDIA GeForce RTX 4090", vram_gb: 24, category: "NVIDIA RTX 40 Series" },
    { value: "RTX-4080-Super", label: "NVIDIA GeForce RTX 4080 Super", vram_gb: 16, category: "NVIDIA RTX 40 Series" },
    { value: "RTX-4080", label: "NVIDIA GeForce RTX 4080", vram_gb: 16, category: "NVIDIA RTX 40 Series" },
    { value: "RTX-4070-Ti-S", label: "NVIDIA GeForce RTX 4070 Ti Super", vram_gb: 16, category: "NVIDIA RTX 40 Series" },
    { value: "RTX-4070-Ti", label: "NVIDIA GeForce RTX 4070 Ti", vram_gb: 12, category: "NVIDIA RTX 40 Series" },
    { value: "RTX-4070-Super", label: "NVIDIA GeForce RTX 4070 Super", vram_gb: 12, category: "NVIDIA RTX 40 Series" },
    { value: "RTX-4070", label: "NVIDIA GeForce RTX 4070", vram_gb: 12, category: "NVIDIA RTX 40 Series" },
    { value: "RTX-4060-Ti-16", label: "NVIDIA GeForce RTX 4060 Ti 16GB", vram_gb: 16, category: "NVIDIA RTX 40 Series" },
    { value: "RTX-4060-Ti", label: "NVIDIA GeForce RTX 4060 Ti", vram_gb: 8, category: "NVIDIA RTX 40 Series" },
    { value: "RTX-4060", label: "NVIDIA GeForce RTX 4060", vram_gb: 8, category: "NVIDIA RTX 40 Series" },

    // ── NVIDIA RTX 30 Series ──
    { value: "RTX-3090-Ti", label: "NVIDIA GeForce RTX 3090 Ti", vram_gb: 24, category: "NVIDIA RTX 30 Series" },
    { value: "RTX-3090", label: "NVIDIA GeForce RTX 3090", vram_gb: 24, category: "NVIDIA RTX 30 Series" },
    { value: "RTX-3080-Ti", label: "NVIDIA GeForce RTX 3080 Ti", vram_gb: 12, category: "NVIDIA RTX 30 Series" },
    { value: "RTX-3080-12GB", label: "NVIDIA GeForce RTX 3080 12GB", vram_gb: 12, category: "NVIDIA RTX 30 Series" },
    { value: "RTX-3080", label: "NVIDIA GeForce RTX 3080", vram_gb: 10, category: "NVIDIA RTX 30 Series" },
    { value: "RTX-3070-Ti", label: "NVIDIA GeForce RTX 3070 Ti", vram_gb: 8, category: "NVIDIA RTX 30 Series" },
    { value: "RTX-3070", label: "NVIDIA GeForce RTX 3070", vram_gb: 8, category: "NVIDIA RTX 30 Series" },
    { value: "RTX-3060-Ti", label: "NVIDIA GeForce RTX 3060 Ti", vram_gb: 8, category: "NVIDIA RTX 30 Series" },
    { value: "RTX-3060", label: "NVIDIA GeForce RTX 3060", vram_gb: 12, category: "NVIDIA RTX 30 Series" },

    // ── NVIDIA Workstation ──
    { value: "RTX-6000-Ada", label: "NVIDIA RTX 6000 Ada", vram_gb: 48, category: "NVIDIA Workstation" },
    { value: "RTX-5000-Ada", label: "NVIDIA RTX 5000 Ada", vram_gb: 32, category: "NVIDIA Workstation" },
    { value: "RTX-4000-Ada", label: "NVIDIA RTX 4000 Ada", vram_gb: 20, category: "NVIDIA Workstation" },
    { value: "RTX-A6000", label: "NVIDIA RTX A6000", vram_gb: 48, category: "NVIDIA Workstation" },
    { value: "RTX-A5000", label: "NVIDIA RTX A5000", vram_gb: 24, category: "NVIDIA Workstation" },
    { value: "RTX-A4000", label: "NVIDIA RTX A4000", vram_gb: 16, category: "NVIDIA Workstation" },

    // ── AMD Data Center ──
    { value: "MI300X", label: "AMD Instinct MI300X", vram_gb: 192, category: "AMD Data Center" },
    { value: "MI250X", label: "AMD Instinct MI250X", vram_gb: 128, category: "AMD Data Center" },
    { value: "MI210", label: "AMD Instinct MI210", vram_gb: 64, category: "AMD Data Center" },

    // ── AMD Consumer ──
    { value: "RX-7900-XTX", label: "AMD Radeon RX 7900 XTX", vram_gb: 24, category: "AMD Consumer" },
    { value: "RX-7900-XT", label: "AMD Radeon RX 7900 XT", vram_gb: 20, category: "AMD Consumer" },
];

export function getVramForModel(model: string): number | null {
    const entry = GPU_MODELS.find((g) => g.value === model);
    return entry ? entry.vram_gb : null;
}

export function getGpusByCategory(): Record<GpuCategory, GpuModel[]> {
    const grouped = {} as Record<GpuCategory, GpuModel[]>;
    for (const cat of GPU_CATEGORIES) {
        grouped[cat] = GPU_MODELS.filter((g) => g.category === cat);
    }
    return grouped;
}
