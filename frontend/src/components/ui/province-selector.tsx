"use client";

import { cn } from "@/lib/utils";
import { Select } from "@/components/ui/input";

const PROVINCES = [
  { code: "all", name: "All Provinces" },
  { code: "AB", name: "Alberta" },
  { code: "BC", name: "British Columbia" },
  { code: "MB", name: "Manitoba" },
  { code: "NB", name: "New Brunswick" },
  { code: "NL", name: "Newfoundland and Labrador" },
  { code: "NS", name: "Nova Scotia" },
  { code: "NT", name: "Northwest Territories" },
  { code: "NU", name: "Nunavut" },
  { code: "ON", name: "Ontario" },
  { code: "PE", name: "Prince Edward Island" },
  { code: "QC", name: "Quebec" },
  { code: "SK", name: "Saskatchewan" },
  { code: "YT", name: "Yukon" },
] as const;

interface ProvinceSelectorProps {
  value: string;
  onChange: (province: string) => void;
  showAll?: boolean;
  className?: string;
}

export function ProvinceSelector({ value, onChange, showAll = true, className }: ProvinceSelectorProps) {
  const options = showAll ? PROVINCES : PROVINCES.filter((p) => p.code !== "all");

  return (
    <Select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className={cn(className)}
    >
      {options.map((p) => (
        <option key={p.code} value={p.code}>
          {p.name}
        </option>
      ))}
    </Select>
  );
}
