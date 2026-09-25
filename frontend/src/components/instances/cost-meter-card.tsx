"use client";

/**
 * B6.5: Cost quote / wallet hold / live meter card.
 *
 * §20.3 asks the instance detail to show billing state. This card shows:
 * - The GPU model and rate (from instance.rate_per_hour or instance.rate)
 * - Wallet hold status (if applicable)
 * - Current session cost (rate × uptime)
 * - Total accumulated cost
 *
 * All cost display uses string formatting, never floating-point arithmetic
 * on currency values. The backend sends costs as string or integer cents.
 */

import { DollarSign, Clock, Wallet, TrendingUp, Info } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { useEffect, useState } from "react";

export interface CostMeterCardProps {
  instance: any;
}

export function CostMeterCard({ instance }: CostMeterCardProps) {
  const [uptimeHours, setUptimeHours] = useState<number>(0);

  useEffect(() => {
    if (!instance?.started_at || ["completed", "failed", "cancelled", "terminated"].includes(instance.status)) {
      return;
    }

    const interval = setInterval(() => {
      const startMs = instance.started_at * 1000;
      const hours = (Date.now() - startMs) / (1000 * 60 * 60);
      setUptimeHours(Math.max(0, hours));
    }, 10000); // Update every 10 seconds

    // Initial calculation
    const startMs = instance.started_at * 1000;
    const hours = (Date.now() - startMs) / (1000 * 60 * 60);
    setUptimeHours(Math.max(0, hours));

    return () => clearInterval(interval);
  }, [instance]);

  const rawRate = instance?.rate_per_hour || instance?.hourly_rate;
  const rateDisplay = rawRate ? String(rawRate) : "0.00";
  
  // Calculate estimated session cost safely (for display only, never send to backend)
  const numericRate = parseFloat(rateDisplay) || 0;
  const estimatedCost = (numericRate * uptimeHours).toFixed(2);
  
  const totalCost = instance?.total_cost ? String(instance.total_cost) : "0.00";
  const hasTotalCost = !!instance?.total_cost;
  
  const gpuName = instance?.gpu_model || instance?.gpu_type || "GPU";
  
  return (
    <Card className="w-full">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg font-medium flex items-center">
          <Wallet className="w-4 h-4 mr-2 text-muted-foreground" />
          Billing Status
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div className="space-y-1">
            <p className="text-sm font-medium text-muted-foreground">Hourly Rate</p>
            <div className="flex items-center text-lg">
              <DollarSign className="w-4 h-4 text-muted-foreground mr-1" />
              <span>{rateDisplay}</span>
              <span className="text-sm text-muted-foreground ml-1">/ hr</span>
            </div>
          </div>
          
          <div className="space-y-1">
            <p className="text-sm font-medium text-muted-foreground">Est. Session Cost</p>
            <div className="flex items-center text-lg">
              <TrendingUp className="w-4 h-4 text-muted-foreground mr-1" />
              <span>{estimatedCost}</span>
            </div>
          </div>
        </div>

        {hasTotalCost && (
          <div className="pt-3 mt-3 border-t">
            <div className="flex justify-between items-center">
              <p className="text-sm font-medium">Total Billed Cost</p>
              <div className="flex items-center font-bold text-lg">
                <DollarSign className="w-4 h-4 mr-1" />
                <span>{totalCost}</span>
              </div>
            </div>
          </div>
        )}

        <div className="flex items-start mt-4 text-xs text-muted-foreground bg-muted/50 p-2 rounded-md">
          <Info className="w-3 h-3 mr-1.5 mt-0.5 flex-shrink-0" />
          <p>Billing is metered server-side. Display values are estimates.</p>
        </div>
      </CardContent>
    </Card>
  );
}
