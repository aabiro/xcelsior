"use client";

import { useState } from "react";
import { cn } from "@/lib/utils";
import { Input, Label } from "@/components/ui/input";
import { Calendar } from "lucide-react";

interface DateRangePickerProps {
  startDate: string;
  endDate: string;
  onStartChange: (date: string) => void;
  onEndChange: (date: string) => void;
  className?: string;
}

export function DateRangePicker({
  startDate,
  endDate,
  onStartChange,
  onEndChange,
  className,
}: DateRangePickerProps) {
  return (
    <div className={cn("flex items-end gap-2", className)}>
      <div className="space-y-1">
        <Label className="text-xs">From</Label>
        <div className="relative">
          <Calendar className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-text-muted pointer-events-none" />
          <Input
            type="date"
            value={startDate}
            onChange={(e) => onStartChange(e.target.value)}
            max={endDate || undefined}
            className="pl-8 text-sm h-9"
          />
        </div>
      </div>
      <span className="pb-2 text-text-muted text-sm">–</span>
      <div className="space-y-1">
        <Label className="text-xs">To</Label>
        <div className="relative">
          <Calendar className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-text-muted pointer-events-none" />
          <Input
            type="date"
            value={endDate}
            onChange={(e) => onEndChange(e.target.value)}
            min={startDate || undefined}
            className="pl-8 text-sm h-9"
          />
        </div>
      </div>
    </div>
  );
}
