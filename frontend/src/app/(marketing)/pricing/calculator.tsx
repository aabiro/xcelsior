"use client";

import { useState } from "react";
import { useLocale } from "@/lib/locale";
import { marketingGpuLabel } from "@/lib/marketing-gpu";

interface GpuOption {
  model: string;
  onDemand: number;
}

export function SavingsCalculator({ gpus }: { gpus: GpuOption[] }) {
  const { t } = useLocale();
  const [selectedGpu, setSelectedGpu] = useState(gpus[0]?.model || "");
  const [hoursPerDay, setHoursPerDay] = useState(8);
  const [daysPerMonth, setDaysPerMonth] = useState(22);

  const gpu = gpus.find((item) => item.model === selectedGpu);
  const rate = gpu?.onDemand ?? 0;
  const totalHours = hoursPerDay * daysPerMonth;
  const monthlyCostOnDemand = totalHours * rate;
  const monthlyCostReserved = monthlyCostOnDemand * 0.55;
  const savings = monthlyCostOnDemand - monthlyCostReserved;

  return (
    <div className="site-calculator">
      <div className="site-calculator-controls">
        <label className="site-field-wrap">
          <span>{t("calc.gpu_model")}</span>
          <select value={selectedGpu} onChange={(event) => setSelectedGpu(event.target.value)} className="site-field">
            {gpus.map((item) => (
              <option key={item.model} value={item.model}>{marketingGpuLabel(item.model)}</option>
            ))}
          </select>
        </label>
        <label className="site-field-wrap">
          <span>{t("calc.hours_day")}</span>
          <input
            type="number"
            min={1}
            max={24}
            value={hoursPerDay}
            onChange={(event) => setHoursPerDay(Math.max(1, Math.min(24, Number(event.target.value))))}
            className="site-field"
          />
        </label>
        <label className="site-field-wrap">
          <span>{t("calc.days_month")}</span>
          <input
            type="number"
            min={1}
            max={31}
            value={daysPerMonth}
            onChange={(event) => setDaysPerMonth(Math.max(1, Math.min(31, Number(event.target.value))))}
            className="site-field"
          />
        </label>
      </div>

      <div className="site-result-grid">
        <ResultBox
          label={t("calc.ondemand_monthly")}
          value={monthlyCostOnDemand}
          sublabel={`${totalHours}h x $${rate.toFixed(2)}/hr`}
        />
        <ResultBox
          label={t("calc.reserved_monthly")}
          value={monthlyCostReserved}
          original={monthlyCostOnDemand}
          strikeLabel={t("calc.reserved_strikethrough_label")}
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
  strikeLabel,
  sublabel,
  highlighted,
  accent,
}: {
  label: string;
  value: number;
  original?: number;
  strikeLabel?: string;
  sublabel: string;
  highlighted?: boolean;
  accent?: boolean;
}) {
  const tone = highlighted ? "gold" : accent ? "green" : "default";

  return (
    <div className="site-result-box" data-tone={tone}>
      <p className="site-result-label">{label}</p>
      {original != null ? (
        <div className="site-result-original">
          {strikeLabel ? <span>{strikeLabel}</span> : null}
          <s>${original.toFixed(2)}</s>
        </div>
      ) : null}
      <p className="site-result-value">${value.toFixed(2)}</p>
      <p className="site-result-note">{sublabel}</p>
    </div>
  );
}
