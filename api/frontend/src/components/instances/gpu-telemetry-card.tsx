"use client";

/**
 * The renter's own GPUs, live and over time.
 *
 * Reads `/api/instances/{id}/telemetry` (newest reading per GPU) and
 * `/telemetry/history` (one point per GPU every 30 s). Both return only the
 * GPUs this instance was given, so on a shared host nothing here describes
 * another tenant. While the instance runs, the newest reading refreshes every
 * 15 s and the chart appends new points every 30 s; a finished run shows its
 * history once, with averages and peaks instead of live tiles.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Activity, AlertTriangle } from "lucide-react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "@/lib/recharts";
import { Card } from "@/components/ui/card";
import {
  ApiError,
  fetchInstanceTelemetry,
  fetchInstanceTelemetryHistory,
  type InstanceGpuHistory,
  type InstanceGpuHistoryPoint,
  type InstanceGpuSample,
  type InstanceTelemetryResponse,
} from "@/lib/api";
import { formatDuration } from "@/lib/format-units";
import { useLocale } from "@/lib/locale";
import { cn } from "@/lib/utils";

const LATEST_EVERY_MS = 15_000;
const HISTORY_EVERY_MS = 30_000;
const HISTORY_WINDOW = 720; // 6 h at one point per 30 s
const MAX_POINTS = 2880;

type ChartPoint = InstanceGpuHistoryPoint;

function mergePoints(
  current: InstanceGpuHistory | null,
  incoming: InstanceGpuHistory,
): InstanceGpuHistory {
  if (!current) return incoming;
  const gpus = incoming.gpus.map((gpu) => {
    const before = current.gpus.find((g) => g.index === gpu.index)?.points ?? [];
    const last = before.length ? before[before.length - 1].t : -Infinity;
    const points = [...before, ...gpu.points.filter((p) => p.t > last)].slice(-MAX_POINTS);
    return { ...gpu, points };
  });
  return { ...incoming, gpus };
}

/** Points with a break wherever the host went quiet, so the line does not bridge the gap. */
function withGaps(points: InstanceGpuHistoryPoint[], interval: number): ChartPoint[] {
  const out: ChartPoint[] = [];
  for (let i = 0; i < points.length; i += 1) {
    const prev = points[i - 1];
    if (prev && points[i].t - prev.t > interval * 3) {
      out.push({ t: (prev.t + points[i].t) / 2, utilization: null, memory_used_gb: null, temp: null, power_draw_w: null });
    }
    out.push(points[i]);
  }
  return out;
}

function num(value: number | null | undefined, digits = 0): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function Tile({ label, value, unit, detail, meter }: {
  label: string;
  value: string;
  unit?: string;
  detail?: string;
  meter?: number | null;
}) {
  return (
    <div className="rounded-lg border border-border/60 bg-surface-hover/50 px-3 py-2.5">
      <p className="text-xs text-text-muted">{label}</p>
      <p className="mt-0.5 font-mono text-lg font-medium text-text-primary">
        {value}
        {unit && value !== "—" && (
          <span className={cn("text-sm text-text-muted", unit !== "%" && "ml-1")}>{unit}</span>
        )}
      </p>
      {typeof meter === "number" && Number.isFinite(meter) && (
        <div className="mt-1.5 h-1 rounded-full bg-border/60" aria-hidden>
          <div
            className="h-1 rounded-full bg-accent-cyan"
            style={{ width: `${Math.min(100, Math.max(0, meter))}%` }}
          />
        </div>
      )}
      {detail && <p className="mt-1 text-xs text-text-muted">{detail}</p>}
    </div>
  );
}

function stats(points: InstanceGpuHistoryPoint[]) {
  const values = (key: keyof InstanceGpuHistoryPoint) =>
    points.map((p) => p[key]).filter((v): v is number => typeof v === "number" && Number.isFinite(v));
  const util = values("utilization");
  const max = (xs: number[]) => (xs.length ? Math.max(...xs) : null);
  return {
    avgUtil: util.length ? util.reduce((a, b) => a + b, 0) / util.length : null,
    peakUtil: max(util),
    peakMemory: max(values("memory_used_gb")),
    peakPower: max(values("power_draw_w")),
    peakTemp: max(values("temp")),
  };
}

export function GpuTelemetryCard({ jobId, live }: { jobId: string; live: boolean }) {
  const { t, locale } = useLocale();
  const [latest, setLatest] = useState<InstanceTelemetryResponse | null>(null);
  const [history, setHistory] = useState<InstanceGpuHistory | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [hidden, setHidden] = useState(false);
  const [selected, setSelected] = useState<number | null>(null);
  const [now, setNow] = useState(() => Date.now() / 1000);
  const historyRef = useRef<InstanceGpuHistory | null>(null);
  useEffect(() => {
    historyRef.current = history;
  }, [history]);

  const loadLatest = useCallback(async () => {
    try {
      const r = await fetchInstanceTelemetry(jobId);
      setLatest(r);
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) setHidden(true);
    }
    setNow(Date.now() / 1000);
  }, [jobId]);

  const loadHistory = useCallback(async (incremental: boolean) => {
    const current = historyRef.current;
    const lastT = incremental && current
      ? Math.max(0, ...current.gpus.flatMap((g) => g.points.map((p) => p.t)))
      : undefined;
    try {
      const r = await fetchInstanceTelemetryHistory(jobId, {
        since: lastT ? lastT + 0.001 : undefined,
        limit: HISTORY_WINDOW,
      });
      if (r.history) {
        const incoming = r.history;
        setHistory((prev) => (incremental ? mergePoints(prev, incoming) : incoming));
      }
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) setHidden(true);
    }
  }, [jobId]);

  useEffect(() => {
    let cancelled = false;
    Promise.allSettled([loadLatest(), loadHistory(false)]).then(() => {
      if (!cancelled) setLoaded(true);
    });
    if (!live) return () => { cancelled = true; };
    const visible = () => typeof document === "undefined" || !document.hidden;
    const latestTimer = setInterval(() => { if (visible()) void loadLatest(); }, LATEST_EVERY_MS);
    const historyTimer = setInterval(() => { if (visible()) void loadHistory(true); }, HISTORY_EVERY_MS);
    return () => {
      cancelled = true;
      clearInterval(latestTimer);
      clearInterval(historyTimer);
    };
  }, [live, loadLatest, loadHistory]);

  const gpus: InstanceGpuSample[] = useMemo(() => latest?.telemetry?.gpus ?? [], [latest]);
  const seriesList = useMemo(() => history?.gpus ?? [], [history]);
  const indexes = useMemo(() => {
    const seen = new Set<number>();
    for (const g of [...gpus, ...seriesList]) if (typeof g.index === "number") seen.add(g.index);
    return [...seen].sort((a, b) => a - b);
  }, [gpus, seriesList]);
  const active = selected !== null && indexes.includes(selected) ? selected : (indexes[0] ?? null);
  const sample = gpus.find((g) => g.index === active) ?? gpus[0] ?? null;
  const series = seriesList.find((g) => g.index === active) ?? seriesList[0] ?? null;
  const points = useMemo(() => series?.points ?? [], [series]);
  const interval = history?.interval_seconds ?? 30;
  const chartData = useMemo(() => withGaps(points, interval), [points, interval]);
  const summary = useMemo(() => stats(points), [points]);
  const hasReadings = points.length > 0 || gpus.some((g) => g.received_at !== null);
  const timeLocale = locale === "fr" ? "fr-CA" : "en-CA";
  const clock = useCallback(
    (epoch: number) =>
      new Date(epoch * 1000).toLocaleTimeString(timeLocale, { hour: "2-digit", minute: "2-digit" }),
    [timeLocale],
  );

  if (hidden) return null;
  if (!loaded) {
    return (
      <Card data-testid="gpu-telemetry-card" aria-busy="true">
        <div className="h-5 w-40 animate-pulse rounded bg-border/60" />
        <div className="mt-4 h-40 animate-pulse rounded-lg bg-border/30" />
      </Card>
    );
  }
  // A finished run with nothing recorded (its host predates per-GPU reports): no card.
  if (!hasReadings && !live) return null;

  const model = sample?.gpu_model || series?.gpu_model || "";
  const age = sample?.received_at ? Math.max(0, now - sample.received_at) : null;
  const ageText = age === null
    ? null
    : age < 90
      ? t("dash.gpu_telemetry.last_reading_s", { n: Math.round(age) })
      : t("dash.gpu_telemetry.last_reading_min", { n: Math.round(age / 60) });
  const gradientId = `gpuUtil-${jobId.replace(/[^A-Za-z0-9_-]/g, "")}`;
  const span = points.length > 1 ? points[points.length - 1].t - points[0].t : 0;

  return (
    <Card data-testid="gpu-telemetry-card">
      <div className="mb-3 flex flex-wrap items-center gap-x-3 gap-y-1">
        <Activity className="h-4 w-4 text-accent-cyan" />
        <h2 className="text-sm font-semibold text-text-secondary">{t("dash.gpu_telemetry.title")}</h2>
        {model && <span className="text-xs text-text-muted">{model}</span>}
        {live && sample && ageText && (
          <span
            className={cn(
              "ml-auto inline-flex items-center gap-1 text-xs",
              sample.stale ? "text-text-secondary" : "text-text-muted",
            )}
            data-testid="gpu-telemetry-age"
          >
            {sample.stale
              ? <AlertTriangle className="h-3.5 w-3.5 text-accent-gold" />
              : <span className="h-1.5 w-1.5 rounded-full bg-emerald" aria-hidden />}
            {ageText}
          </span>
        )}
      </div>

      {indexes.length > 1 && (
        <div role="tablist" aria-label={t("dash.gpu_telemetry.title")} className="mb-3 flex flex-wrap gap-1.5">
          {indexes.map((index) => (
            <button
              key={index}
              type="button"
              role="tab"
              aria-selected={index === active}
              onClick={() => setSelected(index)}
              className={cn(
                "rounded-md border px-2.5 py-1 font-mono text-xs transition-colors",
                index === active
                  ? "border-accent-cyan/50 bg-accent-cyan/10 text-text-primary"
                  : "border-border text-text-muted hover:text-text-primary",
              )}
            >
              {t("dash.gpu_telemetry.gpu_label", { index })}
            </button>
          ))}
        </div>
      )}

      {!hasReadings ? (
        <p className="text-sm text-text-muted" data-testid="gpu-telemetry-empty">
          {latest?.reason === "no_gpu"
            ? t("dash.gpu_telemetry.waiting_gpu")
            : t("dash.gpu_telemetry.no_readings")}
        </p>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-2 lg:grid-cols-4" data-testid="gpu-telemetry-tiles">
            {live && sample && sample.received_at !== null ? (
              <>
                <Tile label={t("dash.gpu_telemetry.utilization")} value={num(sample.utilization)} unit="%" meter={sample.utilization} />
                <Tile
                  label={t("dash.gpu_telemetry.memory")}
                  value={num(sample.memory_used_gb, 1)}
                  unit={sample.memory_total_gb ? `/ ${num(sample.memory_total_gb, 0)} GB` : "GB"}
                  meter={
                    sample.memory_used_gb !== null && sample.memory_total_gb
                      ? (sample.memory_used_gb / sample.memory_total_gb) * 100
                      : sample.memory_util
                  }
                />
                <Tile label={t("dash.gpu_telemetry.temperature")} value={num(sample.temp)} unit="°C" />
                <Tile
                  label={t("dash.gpu_telemetry.power")}
                  value={num(sample.power_draw_w)}
                  unit="W"
                  detail={sample.power_limit_w ? t("dash.gpu_telemetry.of_limit", { limit: num(sample.power_limit_w) }) : undefined}
                />
              </>
            ) : (
              <>
                <Tile label={t("dash.gpu_telemetry.avg_util")} value={num(summary.avgUtil)} unit="%" meter={summary.avgUtil} />
                <Tile label={t("dash.gpu_telemetry.peak_util")} value={num(summary.peakUtil)} unit="%" />
                <Tile label={t("dash.gpu_telemetry.peak_memory")} value={num(summary.peakMemory, 1)} unit="GB" />
                <Tile label={t("dash.gpu_telemetry.peak_power")} value={num(summary.peakPower)} unit="W" />
              </>
            )}
          </div>

          <div className="mt-4">
            <div className="mb-1 flex flex-wrap items-baseline justify-between gap-x-3">
              <h3 className="text-xs font-semibold text-text-secondary">{t("dash.gpu_telemetry.chart_title")}</h3>
              {/* A finished run's tiles already carry the average and peak. */}
              {live && points.length > 0 && (
                <p className="text-xs text-text-muted" data-testid="gpu-telemetry-summary">
                  {t("dash.gpu_telemetry.avg_peak", { avg: num(summary.avgUtil), peak: num(summary.peakUtil) })}
                </p>
              )}
            </div>
            {points.length < 2 ? (
              <p className="py-6 text-center text-xs text-text-muted">{t("dash.gpu_telemetry.chart_empty")}</p>
            ) : (
              <>
                <div
                  className="h-44"
                  role="img"
                  aria-label={t("dash.gpu_telemetry.chart_label", {
                    avg: num(summary.avgUtil),
                    peak: num(summary.peakUtil),
                    span: formatDuration(span),
                  })}
                >
                  <ResponsiveContainer width="100%" height="100%" minHeight={160} minWidth={0} debounce={1}>
                    <AreaChart data={chartData} margin={{ top: 6, right: 6, bottom: 0, left: -18 }}>
                      <defs>
                        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="var(--color-accent-cyan)" stopOpacity={0.25} />
                          <stop offset="100%" stopColor="var(--color-accent-cyan)" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid vertical={false} stroke="var(--color-border)" strokeOpacity={0.6} />
                      <XAxis
                        dataKey="t"
                        type="number"
                        scale="time"
                        domain={["dataMin", "dataMax"]}
                        tickFormatter={clock}
                        tick={{ fill: "var(--color-text-muted)", fontSize: 10 }}
                        stroke="var(--color-border)"
                        tickLine={false}
                        minTickGap={48}
                      />
                      <YAxis
                        domain={[0, 100]}
                        ticks={[0, 50, 100]}
                        tickFormatter={(v: number) => `${v}%`}
                        tick={{ fill: "var(--color-text-muted)", fontSize: 10 }}
                        axisLine={false}
                        tickLine={false}
                      />
                      <Tooltip
                        cursor={{ stroke: "var(--color-border-light)", strokeWidth: 1 }}
                        isAnimationActive={false}
                        content={({ active, payload }) => {
                          const point = active ? (payload?.[0]?.payload as ChartPoint | undefined) : undefined;
                          if (!point || point.utilization === null) return null;
                          return (
                            <div
                              className="rounded-lg border border-border px-3 py-2 text-xs shadow-lg"
                              style={{ background: "var(--popover-solid, var(--color-surface))" }}
                            >
                              <p className="mb-1 font-mono text-text-muted">{clock(point.t)}</p>
                              <p className="flex items-center gap-1.5 text-text-primary">
                                <span className="h-2 w-2 rounded-full bg-accent-cyan" aria-hidden />
                                {t("dash.gpu_telemetry.utilization")} <span className="ml-auto pl-3 font-mono">{num(point.utilization)}%</span>
                              </p>
                              <p className="flex text-text-secondary">
                                {t("dash.gpu_telemetry.memory")} <span className="ml-auto pl-3 font-mono">{num(point.memory_used_gb, 1)} GB</span>
                              </p>
                              <p className="flex text-text-secondary">
                                {t("dash.gpu_telemetry.temperature")} <span className="ml-auto pl-3 font-mono">{num(point.temp)} °C</span>
                              </p>
                              <p className="flex text-text-secondary">
                                {t("dash.gpu_telemetry.power")} <span className="ml-auto pl-3 font-mono">{num(point.power_draw_w)} W</span>
                              </p>
                            </div>
                          );
                        }}
                      />
                      <Area
                        type="monotone"
                        dataKey="utilization"
                        stroke="var(--color-accent-cyan)"
                        strokeWidth={2}
                        fill={`url(#${gradientId})`}
                        dot={false}
                        activeDot={{ r: 4, stroke: "var(--popover-solid, var(--color-surface))", strokeWidth: 2, fill: "var(--color-accent-cyan)" }}
                        isAnimationActive={false}
                        connectNulls={false}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
                <p className="mt-1 text-xs text-text-muted">
                  {t("dash.gpu_telemetry.chart_caption", { seconds: interval, span: formatDuration(span) })}
                </p>
              </>
            )}
          </div>
        </>
      )}
    </Card>
  );
}
