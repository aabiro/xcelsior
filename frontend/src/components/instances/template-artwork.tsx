import { useId } from "react";
import { cn } from "@/lib/utils";
import { ProviderLogo } from "@/components/ui/provider-logo";

type TemplateKey =
  | "pytorch"
  | "tensorflow"
  | "vllm"
  | "comfyui"
  | "jupyter"
  | "ubuntu";

type ThemePalette = {
  bgA: string;
  bgB: string;
  bgC: string;
  glowA: string;
  glowB: string;
  glowC: string;
  majorGrid: string;
  minorGrid: string;
  lineA: string;
  lineB: string;
  lineC: string;
  frame: string;
  panel: string;
};

const TEMPLATE_ART: Record<
  TemplateKey,
  {
    label: string;
    light: ThemePalette;
    dark: ThemePalette;
  }
> = {
  pytorch: {
    label: "PyTorch",
    light: {
      bgA: "#fff4ee",
      bgB: "#eef7ff",
      bgC: "#f8efff",
      glowA: "rgba(249, 115, 22, 0.22)",
      glowB: "rgba(14, 165, 233, 0.18)",
      glowC: "rgba(124, 58, 237, 0.16)",
      majorGrid: "rgba(203, 213, 225, 0.38)",
      minorGrid: "rgba(226, 232, 240, 0.55)",
      lineA: "#f97316",
      lineB: "#0ea5e9",
      lineC: "#7c3aed",
      frame: "rgba(255,255,255,0.78)",
      panel: "rgba(255,255,255,0.72)",
    },
    dark: {
      bgA: "#1a100d",
      bgB: "#0b1522",
      bgC: "#180d22",
      glowA: "rgba(249, 115, 22, 0.26)",
      glowB: "rgba(0, 212, 255, 0.18)",
      glowC: "rgba(124, 58, 237, 0.18)",
      majorGrid: "rgba(43, 55, 78, 0.46)",
      minorGrid: "rgba(26, 34, 53, 0.7)",
      lineA: "#fb923c",
      lineB: "#00d4ff",
      lineC: "#8b5cf6",
      frame: "rgba(255,255,255,0.12)",
      panel: "rgba(9,17,32,0.72)",
    },
  },
  tensorflow: {
    label: "TensorFlow",
    light: {
      bgA: "#fff5ed",
      bgB: "#fffdf8",
      bgC: "#eef6ff",
      glowA: "rgba(234, 88, 12, 0.18)",
      glowB: "rgba(245, 158, 11, 0.16)",
      glowC: "rgba(14, 165, 233, 0.16)",
      majorGrid: "rgba(203, 213, 225, 0.36)",
      minorGrid: "rgba(226, 232, 240, 0.54)",
      lineA: "#ea580c",
      lineB: "#f59e0b",
      lineC: "#0ea5e9",
      frame: "rgba(255,255,255,0.78)",
      panel: "rgba(255,255,255,0.72)",
    },
    dark: {
      bgA: "#1a1109",
      bgB: "#181007",
      bgC: "#0d1622",
      glowA: "rgba(249, 115, 22, 0.26)",
      glowB: "rgba(245, 158, 11, 0.18)",
      glowC: "rgba(0, 212, 255, 0.16)",
      majorGrid: "rgba(43, 55, 78, 0.44)",
      minorGrid: "rgba(26, 34, 53, 0.68)",
      lineA: "#f97316",
      lineB: "#fbbf24",
      lineC: "#38bdf8",
      frame: "rgba(255,255,255,0.12)",
      panel: "rgba(9,17,32,0.72)",
    },
  },
  vllm: {
    label: "vLLM",
    light: {
      bgA: "#eef7ff",
      bgB: "#f8f2ff",
      bgC: "#f5fbff",
      glowA: "rgba(14, 165, 233, 0.22)",
      glowB: "rgba(124, 58, 237, 0.18)",
      glowC: "rgba(220, 38, 38, 0.12)",
      majorGrid: "rgba(203, 213, 225, 0.38)",
      minorGrid: "rgba(226, 232, 240, 0.56)",
      lineA: "#0ea5e9",
      lineB: "#7c3aed",
      lineC: "#dc2626",
      frame: "rgba(255,255,255,0.78)",
      panel: "rgba(255,255,255,0.72)",
    },
    dark: {
      bgA: "#091522",
      bgB: "#140d24",
      bgC: "#0a1220",
      glowA: "rgba(0, 212, 255, 0.22)",
      glowB: "rgba(124, 58, 237, 0.2)",
      glowC: "rgba(220, 38, 38, 0.14)",
      majorGrid: "rgba(43, 55, 78, 0.46)",
      minorGrid: "rgba(26, 34, 53, 0.7)",
      lineA: "#00d4ff",
      lineB: "#8b5cf6",
      lineC: "#ef4444",
      frame: "rgba(255,255,255,0.12)",
      panel: "rgba(9,17,32,0.72)",
    },
  },
  comfyui: {
    label: "ComfyUI",
    light: {
      bgA: "#eef4ff",
      bgB: "#f9f0ff",
      bgC: "#f5fbff",
      glowA: "rgba(37, 99, 235, 0.2)",
      glowB: "rgba(124, 58, 237, 0.18)",
      glowC: "rgba(245, 158, 11, 0.12)",
      majorGrid: "rgba(203, 213, 225, 0.38)",
      minorGrid: "rgba(226, 232, 240, 0.56)",
      lineA: "#2563eb",
      lineB: "#7c3aed",
      lineC: "#f59e0b",
      frame: "rgba(255,255,255,0.78)",
      panel: "rgba(255,255,255,0.72)",
    },
    dark: {
      bgA: "#0a1430",
      bgB: "#160d28",
      bgC: "#091320",
      glowA: "rgba(37, 99, 235, 0.24)",
      glowB: "rgba(124, 58, 237, 0.2)",
      glowC: "rgba(245, 158, 11, 0.14)",
      majorGrid: "rgba(43, 55, 78, 0.46)",
      minorGrid: "rgba(26, 34, 53, 0.7)",
      lineA: "#60a5fa",
      lineB: "#8b5cf6",
      lineC: "#fbbf24",
      frame: "rgba(255,255,255,0.12)",
      panel: "rgba(9,17,32,0.72)",
    },
  },
  jupyter: {
    label: "Jupyter",
    light: {
      bgA: "#fff6ef",
      bgB: "#f5fbff",
      bgC: "#f8efff",
      glowA: "rgba(249, 115, 22, 0.2)",
      glowB: "rgba(14, 165, 233, 0.16)",
      glowC: "rgba(124, 58, 237, 0.14)",
      majorGrid: "rgba(203, 213, 225, 0.38)",
      minorGrid: "rgba(226, 232, 240, 0.56)",
      lineA: "#f97316",
      lineB: "#0ea5e9",
      lineC: "#7c3aed",
      frame: "rgba(255,255,255,0.78)",
      panel: "rgba(255,255,255,0.72)",
    },
    dark: {
      bgA: "#1b100b",
      bgB: "#091420",
      bgC: "#160d22",
      glowA: "rgba(249, 115, 22, 0.24)",
      glowB: "rgba(0, 212, 255, 0.18)",
      glowC: "rgba(124, 58, 237, 0.16)",
      majorGrid: "rgba(43, 55, 78, 0.46)",
      minorGrid: "rgba(26, 34, 53, 0.7)",
      lineA: "#fb923c",
      lineB: "#00d4ff",
      lineC: "#8b5cf6",
      frame: "rgba(255,255,255,0.12)",
      panel: "rgba(9,17,32,0.72)",
    },
  },
  ubuntu: {
    label: "Ubuntu",
    light: {
      bgA: "#fff5ef",
      bgB: "#fffaf6",
      bgC: "#f2f8ff",
      glowA: "rgba(234, 88, 12, 0.18)",
      glowB: "rgba(220, 38, 38, 0.16)",
      glowC: "rgba(14, 165, 233, 0.14)",
      majorGrid: "rgba(203, 213, 225, 0.38)",
      minorGrid: "rgba(226, 232, 240, 0.56)",
      lineA: "#ea580c",
      lineB: "#dc2626",
      lineC: "#0ea5e9",
      frame: "rgba(255,255,255,0.78)",
      panel: "rgba(255,255,255,0.72)",
    },
    dark: {
      bgA: "#1b110b",
      bgB: "#170c12",
      bgC: "#091420",
      glowA: "rgba(249, 115, 22, 0.24)",
      glowB: "rgba(220, 38, 38, 0.18)",
      glowC: "rgba(0, 212, 255, 0.14)",
      majorGrid: "rgba(43, 55, 78, 0.46)",
      minorGrid: "rgba(26, 34, 53, 0.7)",
      lineA: "#fb923c",
      lineB: "#ef4444",
      lineC: "#00d4ff",
      frame: "rgba(255,255,255,0.12)",
      panel: "rgba(9,17,32,0.72)",
    },
  },
};

function renderMotif(template: TemplateKey, palette: ThemePalette) {
  switch (template) {
    case "pytorch":
      return (
        <>
          <path d="M28 72C26 55 33 39 48 29C58 22 69 20 80 24" stroke={palette.lineA} strokeWidth="2.6" strokeLinecap="round" />
          <path d="M24 64C35 82 56 89 77 82C84 79 89 75 92 69" stroke={palette.lineB} strokeWidth="2.2" strokeLinecap="round" opacity="0.86" />
          <path d="M61 18L82 33" stroke={palette.lineC} strokeWidth="2.3" strokeLinecap="round" />
          <circle cx="80" cy="24" r="4.8" fill={palette.lineB} />
          <circle cx="27" cy="72" r="4.4" fill={palette.lineA} />
          <circle cx="83" cy="68" r="3.4" fill={palette.lineC} />
        </>
      );
    case "tensorflow":
      return (
        <>
          <path d="M20 24H80" stroke={palette.lineA} strokeWidth="7" strokeLinecap="round" />
          <path d="M35 24V72" stroke={palette.lineA} strokeWidth="7" strokeLinecap="round" />
          <path d="M52 35H81" stroke={palette.lineB} strokeWidth="6" strokeLinecap="round" opacity="0.9" />
          <path d="M66.5 35V78" stroke={palette.lineB} strokeWidth="6" strokeLinecap="round" opacity="0.9" />
          <path d="M22 78L88 42" stroke={palette.lineC} strokeWidth="2.4" strokeLinecap="round" opacity="0.75" />
          <circle cx="86" cy="42" r="4.2" fill={palette.lineC} />
        </>
      );
    case "vllm":
      return (
        <>
          <path d="M14 33H52C58 33 62 37 62 43C62 49 66 52 72 52H92" stroke={palette.lineA} strokeWidth="2.4" strokeLinecap="round" />
          <path d="M14 51H44C51 51 56 55 56 62C56 68 61 72 68 72H92" stroke={palette.lineB} strokeWidth="2.4" strokeLinecap="round" />
          <path d="M14 69H36C42 69 46 66 48 61" stroke={palette.lineC} strokeWidth="2.2" strokeLinecap="round" />
          <rect x="66" y="21" width="20" height="16" rx="6" fill={palette.panel} stroke={palette.lineB} strokeWidth="2" />
          <path d="M72 37L68 42H76" stroke={palette.lineB} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          <circle cx="50" cy="33" r="3.8" fill={palette.lineA} />
          <circle cx="56" cy="62" r="3.8" fill={palette.lineB} />
          <circle cx="48" cy="61" r="3.2" fill={palette.lineC} />
        </>
      );
    case "comfyui":
      return (
        <>
          <path d="M22 30L43 24L63 38L81 26" stroke={palette.lineA} strokeWidth="2.2" strokeLinecap="round" />
          <path d="M22 30L34 58L57 54L72 73" stroke={palette.lineB} strokeWidth="2.2" strokeLinecap="round" />
          <path d="M43 24L57 54L81 26" stroke={palette.lineC} strokeWidth="2" strokeLinecap="round" opacity="0.82" />
          <circle cx="22" cy="30" r="5" fill={palette.lineA} />
          <circle cx="43" cy="24" r="5" fill={palette.lineB} />
          <circle cx="63" cy="38" r="5" fill={palette.lineC} />
          <circle cx="81" cy="26" r="5" fill={palette.lineA} />
          <circle cx="34" cy="58" r="5" fill={palette.lineC} />
          <circle cx="57" cy="54" r="5" fill={palette.lineB} />
          <circle cx="72" cy="73" r="5" fill={palette.lineA} />
        </>
      );
    case "jupyter":
      return (
        <>
          <rect x="22" y="24" width="50" height="38" rx="9" fill={palette.panel} stroke={palette.lineB} strokeWidth="2" />
          <path d="M31 35H63" stroke={palette.lineA} strokeWidth="2.2" strokeLinecap="round" />
          <path d="M31 43H57" stroke={palette.lineC} strokeWidth="2" strokeLinecap="round" />
          <path d="M31 51H53" stroke={palette.lineA} strokeWidth="2" strokeLinecap="round" opacity="0.75" />
          <path d="M20 62C27 73 39 80 52 80C65 80 77 74 84 64" stroke={palette.lineA} strokeWidth="2.4" strokeLinecap="round" />
          <path d="M26 18C34 11 43 8 53 8C63 8 72 12 79 20" stroke={palette.lineB} strokeWidth="2.4" strokeLinecap="round" />
          <circle cx="82" cy="23" r="4.2" fill={palette.lineB} />
          <circle cx="18" cy="59" r="4.2" fill={palette.lineA} />
        </>
      );
    case "ubuntu":
      return (
        <>
          <rect x="18" y="23" width="58" height="38" rx="10" fill={palette.panel} stroke={palette.lineA} strokeWidth="2.1" />
          <circle cx="27" cy="32" r="3.3" fill={palette.lineB} />
          <path d="M30 43H43L37 49" stroke={palette.lineC} strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
          <path d="M46 49H63" stroke={palette.lineA} strokeWidth="2.4" strokeLinecap="round" />
          <path d="M64 19C73 24 80 33 82 43" stroke={palette.lineB} strokeWidth="2.4" strokeLinecap="round" />
          <path d="M13 49C15 59 20 67 28 73" stroke={palette.lineC} strokeWidth="2.4" strokeLinecap="round" />
          <circle cx="82" cy="43" r="4.2" fill={palette.lineB} />
          <circle cx="28" cy="73" r="4.2" fill={palette.lineC} />
        </>
      );
  }
}

function TemplateArtSvg({
  template,
  palette,
  idPrefix,
}: {
  template: TemplateKey;
  palette: ThemePalette;
  idPrefix: string;
}) {
  const bgId = `${idPrefix}-bg`;
  const haloAId = `${idPrefix}-halo-a`;
  const haloBId = `${idPrefix}-halo-b`;
  const gridFadeId = `${idPrefix}-grid-fade`;

  return (
    <svg viewBox="0 0 100 100" aria-hidden className="h-full w-full">
      <defs>
        <linearGradient id={bgId} x1="11" y1="8" x2="90" y2="94" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor={palette.bgA} />
          <stop offset="52%" stopColor={palette.bgB} />
          <stop offset="100%" stopColor={palette.bgC} />
        </linearGradient>
        <radialGradient id={haloAId} cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(26 22) rotate(53) scale(42 38)">
          <stop offset="0%" stopColor={palette.glowA} />
          <stop offset="100%" stopColor="transparent" />
        </radialGradient>
        <radialGradient id={haloBId} cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(78 78) rotate(90) scale(34 40)">
          <stop offset="0%" stopColor={palette.glowB} />
          <stop offset="100%" stopColor="transparent" />
        </radialGradient>
        <linearGradient id={gridFadeId} x1="0" y1="8" x2="0" y2="94" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="white" />
          <stop offset="74%" stopColor="white" stopOpacity="0.88" />
          <stop offset="100%" stopColor="white" stopOpacity="0.22" />
        </linearGradient>
      </defs>

      <rect width="100" height="100" rx="28" fill={`url(#${bgId})`} />
      <rect width="100" height="100" rx="28" fill={`url(#${haloAId})`} />
      <rect width="100" height="100" rx="28" fill={`url(#${haloBId})`} />
      <rect x="68" y="12" width="20" height="20" rx="10" fill={palette.glowC} />

      <g opacity="0.88" mask={`url(#${gridFadeId})`}>
        {Array.from({ length: 4 }).map((_, index) => (
          <path
            key={`h-${index}`}
            d={`M10 ${18 + index * 18}H90`}
            stroke={index === 1 || index === 2 ? palette.majorGrid : palette.minorGrid}
            strokeWidth={index === 2 ? 1.5 : 1}
          />
        ))}
        {Array.from({ length: 4 }).map((_, index) => (
          <path
            key={`v-${index}`}
            d={`M${18 + index * 18} 10V90`}
            stroke={index === 1 || index === 2 ? palette.majorGrid : palette.minorGrid}
            strokeWidth={index === 2 ? 1.5 : 1}
          />
        ))}
      </g>

      <g opacity="0.96">{renderMotif(template, palette)}</g>
      <rect x="8" y="8" width="84" height="84" rx="23" fill="none" stroke={palette.frame} strokeWidth="1.2" />
    </svg>
  );
}

export function TemplateArtwork({
  template,
  size = 72,
  className,
}: {
  template: string;
  size?: number;
  className?: string;
}) {
  const normalized = template.trim().toLowerCase().replace(/[_\s]+/g, "-") as TemplateKey;
  const artwork = TEMPLATE_ART[normalized];
  const id = useId().replace(/:/g, "");

  if (!artwork) {
    return (
      <span
        className={cn(
          "inline-flex items-center justify-center rounded-[22px] border border-border/70 bg-background/70",
          className,
        )}
        style={{ width: size, height: size }}
      >
        <ProviderLogo provider={template} size={Math.round(size * 0.4)} />
      </span>
    );
  }

  return (
    <span
      className={cn(
        "relative isolate inline-flex shrink-0 overflow-hidden rounded-[24px] border border-border/70 shadow-[0_14px_36px_rgba(15,23,42,0.08)] dark:shadow-[0_16px_38px_rgba(0,0,0,0.34)]",
        className,
      )}
      style={{ width: size, height: size }}
    >
      <span className="absolute inset-0 dark:hidden">
        <TemplateArtSvg template={normalized} palette={artwork.light} idPrefix={`${id}-light`} />
      </span>
      <span className="absolute inset-0 hidden dark:block">
        <TemplateArtSvg template={normalized} palette={artwork.dark} idPrefix={`${id}-dark`} />
      </span>

      <span className="absolute inset-[14%] rounded-[20px] border border-white/[0.45] bg-white/[0.45] backdrop-blur-[10px] dark:border-white/10 dark:bg-white/[0.05]" />
      <span className="absolute inset-[18%] rounded-[18px] border border-white/30 dark:border-white/[0.08]" />

      <span className="relative z-10 flex h-full w-full items-center justify-center">
        <ProviderLogo
          provider={normalized}
          size={Math.max(20, Math.round(size * 0.34))}
          className="drop-shadow-[0_10px_24px_rgba(15,23,42,0.18)]"
        />
      </span>

      <span className="pointer-events-none absolute left-4 top-4 h-2.5 w-2.5 rounded-full bg-white/65 dark:bg-white/20" />
      <span className="pointer-events-none absolute bottom-4 right-4 h-3 w-3 rounded-full border border-white/[0.55] dark:border-white/[0.18]" />
    </span>
  );
}
