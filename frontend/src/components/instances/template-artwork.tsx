import { useId } from "react";
import { cn } from "@/lib/utils";
import { ProviderLogo } from "@/components/ui/provider-logo";
import * as React from "react";

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
      glowB: "rgba(8, 145, 178, 0.18)",
      glowC: "rgba(91, 33, 182, 0.16)",
      majorGrid: "rgba(203, 213, 225, 0.38)",
      minorGrid: "rgba(226, 232, 240, 0.55)",
      lineA: "#f97316",
      lineB: "#0891b2",
      lineC: "#5b21b6",
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
      glowC: "rgba(8, 145, 178, 0.16)",
      majorGrid: "rgba(203, 213, 225, 0.36)",
      minorGrid: "rgba(226, 232, 240, 0.54)",
      lineA: "#ea580c",
      lineB: "#f59e0b",
      lineC: "#0891b2",
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
      glowA: "rgba(8, 145, 178, 0.22)",
      glowB: "rgba(91, 33, 182, 0.18)",
      glowC: "rgba(234, 88, 12, 0.12)",
      majorGrid: "rgba(203, 213, 225, 0.38)",
      minorGrid: "rgba(226, 232, 240, 0.56)",
      lineA: "#0891b2",
      lineB: "#5b21b6",
      lineC: "#ea580c",
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
      glowB: "rgba(91, 33, 182, 0.18)",
      glowC: "rgba(245, 158, 11, 0.12)",
      majorGrid: "rgba(203, 213, 225, 0.38)",
      minorGrid: "rgba(226, 232, 240, 0.56)",
      lineA: "#2563eb",
      lineB: "#5b21b6",
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
      glowB: "rgba(8, 145, 178, 0.16)",
      glowC: "rgba(91, 33, 182, 0.14)",
      majorGrid: "rgba(203, 213, 225, 0.38)",
      minorGrid: "rgba(226, 232, 240, 0.56)",
      lineA: "#f97316",
      lineB: "#0891b2",
      lineC: "#5b21b6",
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
      glowB: "rgba(234, 88, 12, 0.16)",
      glowC: "rgba(8, 145, 178, 0.14)",
      majorGrid: "rgba(203, 213, 225, 0.38)",
      minorGrid: "rgba(226, 232, 240, 0.56)",
      lineA: "#ea580c",
      lineB: "#ea580c",
      lineC: "#0891b2",
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


// Inline SVGs for each template with solid color and creative gradient
function TemplateLogoSVG({ template, size = 64 }: { template: TemplateKey; size?: number }) {
  const uid = useId();
  const g = (name: string) => `${uid}-${name}`;

  switch (template) {
    case 'tensorflow':
      // Solid orange for T+F, subtle gradient on F
      return (
        <svg width={size} height={size} viewBox="0 0 128 128" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id={g("tf-f")} x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#ffb347" />
              <stop offset="100%" stopColor="#ff6f00" />
            </linearGradient>
          </defs>
          {/* T+F base */}
          <path d="M61.55 128L39.71 115.32V40.55L6.81 59.56l.08-28.32L61.55 0z" fill="#ff6f00" />
          {/* F with gradient */}
          <path d="M66.46 0v128l21.84-12.68V79.31l16.49 9.53-.1-24.63-16.39-9.36v-14.3l32.89 19.01-.08-28.32z" fill={`url(#${g("tf-f")})`} />
        </svg>
      );
    case 'pytorch':
      // Solid orange, gradient on the flame tip
      return (
        <svg width={size} height={size} viewBox="-27 0 310 310" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <radialGradient id={g("pt-flame")} cx="70%" cy="20%" r="60%">
              <stop offset="0%" stopColor="#fff4ee" stopOpacity="0.9" />
              <stop offset="100%" stopColor="#EE4C2C" stopOpacity="1" />
            </radialGradient>
          </defs>
          {/* Main circle */}
          <path d="M218.281,90.106C268.573,140.398,268.573,221.075,218.281,271.716C169.037,322.008,88.011,322.008,37.719,271.716C-12.573,221.424,-12.573,140.398,37.719,90.106L127.825,0L127.825,45.053L119.443,53.435L59.722,113.157C22.003,150.177,22.003,210.947,59.722,248.666C96.742,286.385,157.512,286.385,195.231,248.666C232.95,211.645,232.95,150.876,195.231,113.157L218.281,90.106Z" fill="#EE4C2C" />
          {/* Flame tip with gradient */}
          <circle cx="173.23" cy="67.75" r="16.5" fill={`url(#${g("pt-flame")})`} />
        </svg>
      );
    case 'jupyter':
      // Solid orange ring, gradient on orbiting dot
      return (
        <svg width={size} height={size} viewBox="-22 0 300 300" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <radialGradient id={g("jup-dot")} cx="50%" cy="50%" r="80%">
              <stop offset="0%" stopColor="#fff6ef" />
              <stop offset="100%" stopColor="#F37726" />
            </radialGradient>
          </defs>
          {/* Main ring */}
          <path d="M127.952969,60.4 C175.882898,60.4 218,77.6 239.8,103 C231.337863,80.1 216.1,60.4 196.1,46.5 C176.098839,32.5 152.3,25.1 128,25.1 C103.588124,25.1 79.8,32.5 59.8,46.5 C39.8176362,60.4 24.6,80.1 16.1,103 C37.8984531,77.5 79.8,60.4 128,60.4 Z" fill="#F37726" />
          {/* Orbiting dot with gradient */}
          <circle cx="233" cy="17" r="17" fill={`url(#${g("jup-dot")})`} />
        </svg>
      );
    case 'comfyui':
      // Solid blue background, gradient on yellow path
      return (
        <svg width={size} height={size} viewBox="0 0 84 84" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id={g("comfy-yellow")} x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#F0FF41" />
              <stop offset="100%" stopColor="#FFD600" />
            </linearGradient>
          </defs>
          <rect width="84" height="84" rx="18" fill="#172DD7" />
          <path d="M28.5899 69.2727C27.3242 69.2727 26.303 68.8023 25.637 67.9128C24.9524 66.9989 24.774 65.723 25.1471 64.4133L26.6455 59.1518C26.765 58.7329 26.6818 58.2821 26.4212 57.9336C26.1606 57.5858 25.7529 57.381 25.3198 57.381H21.0116C19.7453 57.381 18.724 56.9112 18.0583 56.0218C17.3738 55.1072 17.1953 53.8314 17.5687 52.5216L22.7163 34.5286L23.2847 32.5517C24.0487 29.869 26.8349 27.6888 29.4966 27.6888H34.6517C35.2668 27.6888 35.8079 27.2787 35.9773 26.6835L37.6821 20.6987C38.4453 18.0187 41.2316 15.8385 43.8933 15.8385L54.9181 15.8189L62.9891 15.8182C64.2551 15.8182 65.2763 16.288 65.942 17.1774C66.6265 18.0913 66.805 19.3672 66.4319 20.6769L64.124 28.7803C63.3611 31.4595 60.5748 33.6391 57.9131 33.6391L46.8637 33.6601H41.7104C41.0959 33.6601 40.5555 34.0695 40.3851 34.6641L36.0883 49.6722C35.9681 50.0919 36.0513 50.5441 36.3126 50.8925C36.5732 51.2403 36.9809 51.445 37.4136 51.445L44.7152 51.4308H52.7622C54.0282 51.4308 55.0494 51.9006 55.7151 52.7901C56.3996 53.7046 56.5781 54.9805 56.2047 56.2902L53.8969 64.3923C53.1339 67.0722 50.3476 69.2517 47.686 69.2517L36.6369 69.2727H28.5899Z" fill={`url(#${g("comfy-yellow")})`} />
        </svg>
      );
    case 'ubuntu':
      // Solid orange background, gradient on white ring
      return (
        <svg width={size} height={size} viewBox="0 0 349 349" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <radialGradient id={g("ubuntu-ring")} cx="50%" cy="50%" r="80%">
              <stop offset="0%" stopColor="#fff" />
              <stop offset="100%" stopColor="#ffe0c2" />
            </radialGradient>
          </defs>
          <rect width="349" height="349" rx="80" fill="#e9500e" />
          <path d="m76 206.3c-20.3 0-36.6-16.4-36.6-36.6 0-20.2 16.3-36.5 36.6-36.5 20.2 0 36.5 16.3 36.5 36.5 0 20.2-16.3 36.6-36.5 36.6zm152.2-80.1c-20.2 0-36.6-16.3-36.6-36.5 0-20.2 16.4-36.6 36.6-36.6 20.2 0 36.5 16.4 36.5 36.6 0 20.2-16.3 36.5-36.5 36.5zm-69.8 137.5c-6.5-1.3-12.7-3.4-18.7-6.2-6-2.7-11.6-6.1-16.9-10.1-5.2-4-10-8.5-14.2-13.6-4.2-5-7.9-10.5-10.9-16.3q-3.7 1.6-7.5 2.7-3.9 1.1-7.8 1.6-4 0.5-8 0.3-4-0.1-8-0.8c3.7 9 8.4 17.5 14.1 25.3 5.7 7.9 12.4 15 19.8 21.2 7.5 6.3 15.6 11.6 24.4 15.8 8.7 4.3 17.9 7.4 27.4 9.4q3.2 0.7 6.4 1.2 3.3 0.5 6.5 0.9 3.3 0.3 6.5 0.5 3.3 0.1 6.6 0.1-2.5-3.3-4.5-6.9-2-3.6-3.3-7.5-1.4-3.9-2.1-8-0.8-4-0.8-8.1-4.6-0.5-9-1.5zm61.5 36.7c-20.2 0-36.5-16.3-36.5-36.5 0-20.2 16.3-36.5 36.5-36.5 20.2 0 36.6 16.3 36.6 36.5 0 20.2-16.4 36.5-36.6 36.5zm50.9-49.9c7.8-9.8 14-20.8 18.3-32.5 4.4-11.8 6.9-24.1 7.5-36.6 0.5-12.5-0.9-25-4.3-37.1-3.4-12.1-8.6-23.5-15.5-34q-1.6 3.7-3.7 7.1-2.1 3.5-4.7 6.5-2.7 3.1-5.7 5.7-3.1 2.6-6.5 4.8c3.7 6.8 6.4 14.1 8.2 21.7 1.7 7.6 2.5 15.4 2.2 23.1-0.2 7.8-1.5 15.5-3.8 22.9-2.2 7.5-5.4 14.6-9.5 21.2q3.2 2.6 5.9 5.7 2.8 3.1 5 6.5 2.2 3.5 3.9 7.3 1.6 3.7 2.7 7.7z" fill={`url(#${g("ubuntu-ring")})`} />
        </svg>
      );
    case 'vllm':
      // Two-tone: gold and blue, gradient on gold wedge
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id={g("vllm-gold")} x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#fffbe6" />
              <stop offset="100%" stopColor="#FDB515" />
            </linearGradient>
          </defs>
          {/* Gold wedge with gradient */}
          <path d="M0 4.973h9.324V23L0 4.973z" fill={`url(#${g("vllm-gold")})`} />
          {/* Blue wedge solid */}
          <path d="M13.986 4.351L22.378 0l-6.216 23H9.324l4.662-18.649z" fill="#30A2FF" />
        </svg>
      );
    default:
      return null;
  }
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

  // Icon sizing: the logo itself is ~50% of the container so there's comfortable padding
  const logoSize = Math.round(size * 0.5);

  if (!artwork) {
    return (
      <span
        className={cn(
          "inline-flex items-center justify-center rounded-[22px] border border-border/70 bg-background/70",
          className,
        )}
        style={{ width: size, height: size }}
      >
        <ProviderLogo provider={template} size={logoSize} />
      </span>
    );
  }

  return (
    <span
      className={cn(
        "relative inline-flex items-center justify-center overflow-hidden rounded-2xl",
        className,
      )}
      style={{ width: size, height: size }}
    >
      {/* Subtle brand-colored glow behind the icon */}
      <span
        className="pointer-events-none absolute inset-0 dark:hidden"
        style={{
          background: `radial-gradient(circle at 50% 50%, ${artwork.light.glowA} 0%, transparent 65%)`,
        }}
      />
      <span
        className="pointer-events-none absolute inset-0 hidden dark:block"
        style={{
          background: `radial-gradient(circle at 50% 50%, ${artwork.dark.glowA} 0%, transparent 65%)`,
        }}
      />
      <TemplateLogoSVG template={normalized} size={logoSize} />
    </span>
  );
}
