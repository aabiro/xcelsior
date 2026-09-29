"use client";

/** Show the instance API's compute estimate; the wallet ledger owns charges. */
import { Wallet, TrendingUp, Info } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { Instance } from "@/lib/api";

export interface CostMeterCardProps {
  instance: Pick<Instance, "rate_per_hour_cad" | "cost_cad" | "rate_is_estimate">;
}

function formatCad(value: number | undefined): string {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? value.toLocaleString("en-CA", { minimumFractionDigits: 2, maximumFractionDigits: 6 })
    : "Unavailable";
}

export function CostMeterCard({ instance }: CostMeterCardProps) {
  return (
    <Card className="w-full">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg font-medium flex items-center">
          <Wallet className="w-4 h-4 mr-2 text-text-muted" />
          Billing Status
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div className="space-y-1">
            <p className="text-sm font-medium text-text-muted">
              {instance.rate_is_estimate ? "Est. Hourly Rate" : "Hourly Rate"}
            </p>
            <div className="flex items-center text-lg">
              <span>{formatCad(instance.rate_per_hour_cad)}</span>
              <span className="text-sm text-text-muted ml-1">CAD / hr</span>
            </div>
          </div>
          <div className="space-y-1">
            <p className="text-sm font-medium text-text-muted">Est. Compute Cost</p>
            <div className="flex items-center text-lg">
              <TrendingUp className="w-4 h-4 text-text-muted mr-1" />
              <span>{formatCad(instance.cost_cad)}</span>
              <span className="text-sm text-text-muted ml-1">CAD</span>
            </div>
          </div>
        </div>
        <div className="flex items-start mt-4 text-xs text-text-muted bg-surface-hover p-2 rounded-md">
          <Info className="w-3 h-3 mr-1.5 mt-0.5 flex-shrink-0" />
          <p>Compute estimate from the latest server update. Final charges appear in billing history.</p>
        </div>
      </CardContent>
    </Card>
  );
}
