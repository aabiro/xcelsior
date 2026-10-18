// ascii-viz.ts — Tiny data-driven ASCII chart renderers for the Learn pane.
//
// Every function is PURE: it takes data and returns plain strings (a single
// line for sparklines, an array of lines for multi-row charts). The rendering
// component wraps the lines in <Text> and applies the brand gradient — keeping
// the geometry here fully unit-testable without Ink (Part C — ASCII data-viz).
//
// Design rules these all honor (the "no bugs" bar):
//   • never divide by zero (flat series, zero max, single point)
//   • clamp to the requested width/height, never overflow the pane
//   • tolerate NaN/Infinity/negative input by coercing to a safe range

const SPARK_TICKS = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"] as const;

/** Coerce to a finite number, falling back to `fallback` (default 0). */
function finite(n: number, fallback = 0): number {
    return Number.isFinite(n) ? n : fallback;
}

/**
 * Render a single-line sparkline from a series of numbers.
 * Empty input → "". A flat series renders as a mid-height baseline.
 */
export function sparkline(values: number[]): string {
    const data = values.map((v) => finite(v));
    if (data.length === 0) return "";
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min;
    return data
        .map((v) => {
            if (range === 0) return SPARK_TICKS[Math.floor((SPARK_TICKS.length - 1) / 2)];
            const t = (v - min) / range;
            const idx = Math.round(t * (SPARK_TICKS.length - 1));
            return SPARK_TICKS[Math.max(0, Math.min(SPARK_TICKS.length - 1, idx))];
        })
        .join("");
}

export interface BarItem {
    label: string;
    value: number;
    /** Optional explicit max for this row; defaults to the max across all rows. */
    max?: number;
}

export interface BarChartOpts {
    /** Width of the bar track in characters (default 24). */
    width?: number;
    /** Label column width; defaults to the longest label (capped at 16). */
    labelWidth?: number;
    /** "percent" shows N%, "value" shows the raw number (default "percent"). */
    valueMode?: "percent" | "value";
    /** Suffix appended to raw values in "value" mode (e.g. "%", " GB"). */
    unit?: string;
}

const BAR_FULL = "█";
const BAR_EMPTY = "░";

/**
 * Horizontal bar chart — one row per item. Used for GPU utilization, funnel
 * conversion, etc. Returns aligned lines like:
 *   "app_launched   ████████████████████████ 100%"
 *   "ride_requested █████████████████░░░░░░░  72%"
 */
export function barChart(items: BarItem[], opts: BarChartOpts = {}): string[] {
    if (items.length === 0) return [];
    const width = Math.max(1, Math.floor(opts.width ?? 24));
    const valueMode = opts.valueMode ?? "percent";
    const unit = opts.unit ?? (valueMode === "percent" ? "%" : "");

    const globalMax = Math.max(1, ...items.map((it) => finite(it.value)));
    const labelWidth = Math.min(
        16,
        opts.labelWidth ?? Math.max(...items.map((it) => it.label.length)),
    );

    return items.map((it) => {
        const value = finite(it.value);
        const max = Math.max(1e-9, finite(it.max ?? globalMax, globalMax));
        const ratio = Math.max(0, Math.min(1, value / max));
        const filled = Math.round(ratio * width);
        const bar = BAR_FULL.repeat(filled) + BAR_EMPTY.repeat(width - filled);
        const label = it.label.length > labelWidth
            ? it.label.slice(0, labelWidth - 1) + "…"
            : it.label.padEnd(labelWidth);
        const valueText = valueMode === "percent"
            ? `${Math.round(ratio * 100)}${unit}`
            : `${trimNumber(value)}${unit}`;
        return `${label} ${bar} ${valueText}`;
    });
}

/** Format a number compactly: 9575 → "9,575", 3.50 → "3.5". */
export function trimNumber(n: number): string {
    const v = finite(n);
    if (Number.isInteger(v)) return v.toLocaleString("en-US");
    return String(Math.round(v * 100) / 100);
}

/** Format a large count compactly for an axis: 9575 → "9.6k", 1200000 → "1.2M". */
export function compactNumber(n: number): string {
    const v = finite(n);
    const abs = Math.abs(v);
    if (abs >= 1_000_000) return `${Math.round(v / 100_000) / 10}M`;
    if (abs >= 1_000) return `${Math.round(v / 100) / 10}k`;
    return trimNumber(v);
}

export interface LineChartOpts {
    /** Plot width in columns (data is sampled/truncated to fit). Default 32. */
    width?: number;
    /** Plot height in rows. Default 6. */
    height?: number;
    /** Show a left y-axis with min/max labels. Default true. */
    axis?: boolean;
}

/**
 * Compact line chart — plots the top-most cell per column so the series reads
 * as a rising/falling line (like the PostHog trends tile). Returns `height`
 * rows; when `axis` is on, each row is prefixed with a right-aligned y label on
 * the top and bottom rows only.
 */
export function lineChart(values: number[], opts: LineChartOpts = {}): string[] {
    const width = Math.max(1, Math.floor(opts.width ?? 32));
    const height = Math.max(1, Math.floor(opts.height ?? 6));
    const axis = opts.axis ?? true;

    const data = values.map((v) => finite(v));
    if (data.length === 0) return Array.from({ length: height }, () => "");

    // Sample the series down to `width` columns (nearest-neighbour).
    const cols: number[] = [];
    for (let x = 0; x < width; x++) {
        const srcIdx = data.length === 1 ? 0 : Math.round((x / (width - 1)) * (data.length - 1));
        cols.push(data[Math.max(0, Math.min(data.length - 1, srcIdx))]);
    }

    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;

    // Row index (0 = top) for each column.
    const rowOf = (v: number) => {
        const t = (v - min) / range; // 0..1 (1 = max = top)
        const row = Math.round((1 - t) * (height - 1));
        return Math.max(0, Math.min(height - 1, row));
    };

    const axisLabels = axis
        ? buildAxisLabels(min, max, height)
        : Array.from({ length: height }, () => "");
    const labelW = axis ? Math.max(...axisLabels.map((l) => l.length)) : 0;

    const rows: string[] = [];
    for (let r = 0; r < height; r++) {
        let line = "";
        for (let x = 0; x < width; x++) {
            const target = rowOf(cols[x]);
            if (target === r) {
                line += "•";
            } else if (x > 0 && between(r, rowOf(cols[x - 1]), target)) {
                // draw a connector when the line crosses this row between columns
                line += "│";
            } else {
                line += " ";
            }
        }
        const prefix = axis ? `${axisLabels[r].padStart(labelW)} ┤` : "";
        rows.push(prefix + line);
    }
    return rows;
}

function between(r: number, a: number, b: number): boolean {
    const lo = Math.min(a, b);
    const hi = Math.max(a, b);
    return r > lo && r < hi;
}

function buildAxisLabels(min: number, max: number, height: number): string[] {
    const labels = Array.from({ length: height }, () => "");
    labels[0] = compactNumber(max);
    labels[height - 1] = compactNumber(min);
    return labels;
}

export interface FunnelStage {
    label: string;
    value: number;
}

/**
 * Conversion funnel — like the PostHog "ride conversion" tile. Each stage shows
 * a bar relative to the first stage plus the percentage and absolute drop.
 */
export function funnel(stages: FunnelStage[], width = 24): string[] {
    if (stages.length === 0) return [];
    const top = Math.max(1, finite(stages[0].value));
    const lines: string[] = [];
    stages.forEach((stage, i) => {
        const value = finite(stage.value);
        const ratio = Math.max(0, Math.min(1, value / top));
        const filled = Math.round(ratio * width);
        const bar = BAR_FULL.repeat(filled) + BAR_EMPTY.repeat(width - filled);
        const pct = `${Math.round(ratio * 100)}%`;
        lines.push(`${i + 1}. ${stage.label}`);
        if (i === 0) {
            lines.push(`   ${bar} ${pct}`);
            lines.push(`   → ${trimNumber(value)}`);
        } else {
            const prev = finite(stages[i - 1].value);
            const drop = prev - value;
            const dropPct = prev > 0 ? Math.round((drop / prev) * 100) : 0;
            lines.push(`   ${bar} ${pct}`);
            lines.push(`   → ${trimNumber(value)}   ↓ ${trimNumber(drop)} (${dropPct}%)`);
        }
    });
    return lines;
}

/**
 * Horizontal pipeline diagram: submit → schedule → run → settle.
 * `activeIndex` highlights the current stage with a filled marker.
 */
export function pipeline(stages: string[], activeIndex = -1): string {
    if (stages.length === 0) return "";
    return stages
        .map((s, i) => (i === activeIndex ? `◆ ${s}` : `◇ ${s}`))
        .join("  →  ");
}
