import Image from "next/image";
import { cn } from "@/lib/utils";

const LOGO_MAP: Record<string, string> = {
  pytorch: "/logos/pytorch.svg",
  tensorflow: "/logos/tensorflow.svg",
  vllm: "/logos/vllm.svg",
  comfyui: "/logos/comfyui.svg",
  jupyter: "/logos/jupyter.svg",
  ubuntu: "/logos/ubuntu.svg",
  aws: "/logos/aws.svg",
  azure: "/logos/azure.svg",
  google: "/logos/google.svg",
  "google-cloud": "/logos/google-cloud.svg",
  googlecloud: "/logos/google-cloud.svg",
  gcp: "/logos/google-cloud.svg",
  github: "/logos/github.svg",
  huggingface: "/logos/huggingface.svg",
  nvidia: "/logos/nvidia.svg",
  runpod: "/logos/runpod.svg",
  vast: "/logos/vast.svg",
};

const LOGO_LABELS: Record<string, string> = {
  pytorch: "PyTorch",
  tensorflow: "TensorFlow",
  vllm: "vLLM",
  comfyui: "ComfyUI",
  jupyter: "Jupyter",
  ubuntu: "Ubuntu",
  aws: "AWS",
  azure: "Azure",
  google: "Google",
  "google-cloud": "Google Cloud",
  googlecloud: "Google Cloud",
  gcp: "Google Cloud",
  github: "GitHub",
  huggingface: "Hugging Face",
  nvidia: "NVIDIA",
  runpod: "RunPod",
  vast: "Vast.ai",
};

const LOGO_IMAGE_CLASSES: Record<string, string> = {
  github: "brightness-0 invert",
};

function normalizeProvider(provider: string): string {
  return provider.trim().toLowerCase().replace(/[_\s]+/g, "-");
}

function resolveLogo(provider: string): { key: string; src: string } | null {
  const normalized = normalizeProvider(provider);
  const compact = normalized.replace(/-/g, "");
  const key = LOGO_MAP[normalized] ? normalized : LOGO_MAP[compact] ? compact : "";
  if (!key) return null;
  return { key, src: LOGO_MAP[key] };
}

export function hasProviderLogo(provider: string): boolean {
  return Boolean(resolveLogo(provider));
}

interface ProviderLogoProps {
  provider: string;
  size?: number;
  framed?: boolean;
  className?: string;
  imageClassName?: string;
  label?: string;
}

export function ProviderLogo({
  provider,
  size = 20,
  framed = false,
  className,
  imageClassName,
  label,
}: ProviderLogoProps) {
  const resolved = resolveLogo(provider);
  if (!resolved) return null;

  const imageSize = framed ? Math.max(16, Math.round(size * 0.58)) : size;
  const image = (
    <Image
      src={resolved.src}
      alt={label ?? LOGO_LABELS[resolved.key] ?? provider}
      width={imageSize}
      height={imageSize}
      className={cn(
        "h-auto w-auto object-contain",
        LOGO_IMAGE_CLASSES[resolved.key],
        !framed && className,
        imageClassName,
      )}
      unoptimized
    />
  );

  if (!framed) return image;

  return (
    <span
      className={cn(
        "inline-flex shrink-0 items-center justify-center rounded-2xl border border-border/70 bg-background/70 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]",
        className,
      )}
      style={{ width: size, height: size }}
    >
      {image}
    </span>
  );
}
