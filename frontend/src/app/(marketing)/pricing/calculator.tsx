"use client";

import { useState } from "react";
import { Select, Input, Label } from "@/components/ui/input";
import { useLocale } from "@/lib/locale";

interface GpuOption {
  model: string;
  onDemand: number;
}

export function SavingsCalculator({ gpus }: { gpus: GpuOption[] }) {
  const { t } = useLocale();
  const [selectedGpu, setSelectedGpu] = useState(gpus[0]?.model || "");
  const [hoursPerDay, setHoursPerDay] = useState(8);
  const [daysPerMonth, setDaysPerMonth] = useState(22);
  const [useRebate, setUseRebate] = useState(true);

  const gpu = gpus.find((g) => g.model === selectedGpu);
  const rate = gpu?.onDemand ?? 0;

  const totalHours = hoursPerDay * daysPerMonth;
  const monthlyCostOnDemand = totalHours * rate;
  const monthlyCostReserved = monthlyCostOnDemand * 0.55; // 45% discount for yearly
  const rebateDiscount = useRebate ? 0.67 : 0;
  const effectiveOnDemand = monthlyCostOnDemand * (1 - rebateDiscount);
  const effectiveReserved = monthlyCostReserved * (1 - rebateDiscount);
  const savings = monthlyCostOnDemand - effectiveReserved;

  return (
    <div className="rounded-xl border border-border bg-surface p-6 md:p-8 max-w-3xl mx-auto">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4 mb-6">
        <div className="space-y-1.5">
          <Label className="text-xs">{t("calc.gpu_model")}</Label>
          <Select value={selectedGpu} onChange={(e) => setSelectedGpu(e.target.value)}>
            {gpus.map((g) => (
              <option key={g.model} value={g.model}>{g.model}</option>
            ))}
          </Select>
        </div>
        <div className="space-y-1.5">
          <Label className="text-xs">{t("calc.hours_day")}</Label>
          <Input
            type="number"
            min={1}
            max={24}
            value={hoursPerDay}
            onChange={(e) => setHoursPerDay(Math.max(1, Math.min(24, Number(e.target.value))))}
          />
        </div>
        <div className="space-y-1.5">
          <Label className="text-xs">{t("calc.days_month")}</Label>
          <Input
            type="number"
            min={1}
            max={31}
            value={daysPerMonth}
            onChange={(e) => setDaysPerMonth(Math.max(1, Math.min(31, Number(e.target.value))))}
          />
        </div>
        <div className="space-y-1.5">
          <Label className="text-xs">{t("calc.fund_toggle")}</Label>
          <button
            onClick={() => setUseRebate(!useRebate)}
            className={`flex h-10 w-full items-center justify-center rounded-lg border text-sm font-medium transition-colors ${
              useRebate
                ? "border-accent-gold bg-accent-gold/10 text-accent-gold"
                : "border-border text-text-muted hover:bg-surface-hover"
            }`}
          >
            {useRebate ? t("calc.fund_on") : t("calc.fund_off")}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <ResultBox
          label={t("calc.ondemand_monthly")}
          value={effectiveOnDemand}
          original={useRebate ? monthlyCostOnDemand : undefined}
          sublabel={`${totalHours}h × $${rate.toFixed(2)}/hr`}
        />
        <ResultBox
          label={t("calc.reserved_monthly")}
          value={effectiveReserved}
          original={useRebate ? monthlyCostReserved : undefined}
          sublabel={t("calc.reserved_note")}
          highlighted
        />
        <ResultBox
          label={t("calc.total_savings")}
          value={savings}
          sublabel={t("calc.savings_note")}
          accent
        />
      </div>
    </div>
  );
}

function ResultBox({
  label,
  value,
  original,
  sublabel,
  highlighted,
  accent,
}: {
  label: string;
  value: number;
  original?: number;
  sublabel: string;
  highlighted?: boolean;
  accent?: boolean;
}) {
  return (
    <div
      className={`rounded-lg border p-4 text-center ${
        highlighted
          ? "border-accent-gold/30 bg-accent-gold/5"
          : accent
            ? "border-emerald/30 bg-emerald/5"
            : "border-border bg-navy-light"
      }`}
    >
      <p className="text-xs text-text-muted mb-1">{label}</p>
      {original != null && (
        <p className="text-sm text-text-muted line-through">${original.toFixed(2)}</p>
      )}
      <p className={`text-2xl font-bold font-mono ${accent ? "text-emerald" : highlighted ? "text-accent-gold" : ""}`}>
        ${value.toFixed(2)}
      </p>
      <p className="text-xs text-text-muted mt-0.5">{sublabel}</p>
    </div>
  );
}
