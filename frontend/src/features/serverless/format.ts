import type { ServerlessPricing } from "@/lib/api";

export function formatModelDisplayName(modelRef?: string | null): string {
  const value = String(modelRef || "").trim();
  if (!value) return "";
  const parts = value.split("/").filter(Boolean);
  return parts[parts.length - 1] || value;
}

export function formatServerlessChip(value?: string | null): string {
  const label = String(value || "").trim().replace(/[_-]+/g, " ");
  if (!label) return "";
  return label.replace(/\w\S*/g, (word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase());
}

export function formatWorkerSecondPrice(pricing?: ServerlessPricing | null): string {
  const rate = pricing?.rate_per_second_cad_per_worker;
  if (rate == null) return "Price: -";
  return `Price: $${rate.toFixed(6)}/worker/sec`;
}
