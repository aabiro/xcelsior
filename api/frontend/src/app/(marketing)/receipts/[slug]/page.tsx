import { notFound } from "next/navigation";
import Link from "next/link";
import { ArrowLeft, ArrowRight } from "lucide-react";
import type { Metadata } from "next";
import { connection } from "next/server";
import { formatDuration, formatMicros } from "@/lib/format-units";
import {
  fetchReceiptPage,
  monthLabel,
  pageTitle,
  pricingModeLabel,
} from "@/lib/receipt-pages";
import { formatHourlyRate } from "@/lib/hourly-rate";

// Three short figures a row: tight enough to fit a phone without scrolling.
const CELL = { padding: "12px 10px" } as const;

interface Props {
  params: Promise<{ slug: string }>;
}

function describe(page: { runs: number; template_label: string; gpu_model: string; median_total_micros: number; median_billed_seconds: number }) {
  return (
    `${page.runs} real ${page.template_label} runs on ${page.gpu_model}: a typical run took ` +
    `${formatDuration(page.median_billed_seconds)} and cost ${formatMicros(page.median_total_micros)}, as billed.`
  );
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  await connection();
  const { slug } = await params;
  const page = await fetchReceiptPage(slug);
  if (!page || page === "missing") return { title: "Workload not found", robots: { index: false } };
  const title = pageTitle(page);
  const description = describe(page);
  return {
    title,
    description,
    alternates: { canonical: `https://xcelsior.ca/receipts/${slug}` },
    openGraph: { title, description, url: `https://xcelsior.ca/receipts/${slug}`, type: "article" },
    twitter: { title, description },
  };
}

export default async function ReceiptPageView({ params }: Props) {
  // Rendered per request so an API blip is never cached as the page; the API
  // answer itself is cached (see fetchReceiptPage).
  await connection();
  const { slug } = await params;
  const page = await fetchReceiptPage(slug);
  if (page === "missing") notFound();
  if (page === null) {
    // The API did not answer: say so rather than pretending the page is gone.
    return (
      <div className="site-container">
        <section className="site-rails site-section site-article-shell">
          <div className="site-empty-state" role="status">
            <p>These figures could not be loaded just now. Try again in a minute.</p>
            <Link href="/receipts" className="site-inline-link">All workload pages</Link>
          </div>
        </section>
      </div>
    );
  }

  const updated = monthLabel(page.latest_month);
  const modes = Object.entries(page.pricing_modes)
    .sort((a, b) => b[1] - a[1])
    .map(([mode, n]) => `${n} ${pricingModeLabel(mode)}`)
    .join(", ");
  const sources =
    page.operator_runs > 0
      ? `${page.customer_runs} shared by customers and ${page.operator_runs} operator ${page.operator_runs === 1 ? "run" : "runs"} on the live fleet`
      : `all shared by the customers who ran them`;
  const rows: [string, string, string, string][] = [
    [
      "Cost",
      formatMicros(page.min_total_micros),
      formatMicros(page.median_total_micros),
      formatMicros(page.max_total_micros),
    ],
    [
      "Billed time",
      formatDuration(page.min_billed_seconds),
      formatDuration(page.median_billed_seconds),
      formatDuration(page.max_billed_seconds),
    ],
  ];

  return (
    <div className="site-container">
      <section className="site-rails site-section site-article-shell">
        <Link href="/receipts" className="site-article-back">
          <ArrowLeft className="site-meta-icon" /> All workload pages
        </Link>

        <article className="site-article" data-testid="receipt-page">
          <header className="site-article-header">
            <div className="site-eyebrow">Real runs · {page.runs} shared receipts{updated ? ` · updated ${updated}` : ""}</div>
            <h1 className="site-article-title">{pageTitle(page)}</h1>
            <p className="site-article-description">
              {page.template_description} These figures come from {page.runs} finished runs of the{" "}
              {page.template_label} quick-start template on {page.gpu_model}, {sources}. Every number is
              what the ledger billed.
            </p>
          </header>

          <div className="site-kpi-strip" style={{ marginTop: 28 }}>
            <div className="site-kpi">
              <div className="site-kpi-label">Typical cost per run</div>
              <div className="site-kpi-value">{formatMicros(page.median_total_micros)}</div>
            </div>
            <div className="site-kpi">
              <div className="site-kpi-label">Typical billed time</div>
              <div className="site-kpi-value">{formatDuration(page.median_billed_seconds)}</div>
            </div>
            <div className="site-kpi">
              {page.median_gpu_utilization !== null ? (
                <>
                  <div className="site-kpi-label">Typical GPU use</div>
                  <div className="site-kpi-value">{Math.round(page.median_gpu_utilization)}%</div>
                </>
              ) : (
                <>
                  <div className="site-kpi-label">Hourly rate</div>
                  <div className="site-kpi-value">
                    {page.median_rate_micros_per_hour !== null
                      ? `${formatHourlyRate(page.median_rate_micros_per_hour / 1_000_000)}/hr`
                      : "—"}
                  </div>
                </>
              )}
            </div>
          </div>

          <div className="site-prose">
            <h2 className="site-prose-h2">The spread</h2>
            <p className="site-prose-p">
              &ldquo;Typical&rdquo; is the median run. Runs differ in how long people keep them going, so
              the range matters as much as the middle.
            </p>
          </div>
          <div className="site-table-wrap" style={{ marginTop: 16 }}>
            <table className="site-table" style={{ minWidth: 0 }}>
              <thead>
                <tr>
                  <th scope="col" style={{ ...CELL, paddingLeft: 16 }}>Per run</th>
                  <th scope="col" style={CELL}>Lowest</th>
                  <th scope="col" style={CELL}>Median</th>
                  <th scope="col" style={CELL}>Highest</th>
                </tr>
              </thead>
              <tbody>
                {rows.map(([label, low, mid, high]) => (
                  <tr key={label}>
                    <th scope="row" style={{ ...CELL, textAlign: "left", background: "transparent", paddingLeft: 16 }}>
                      {label}
                    </th>
                    <td style={{ ...CELL, textAlign: "center" }}>{low}</td>
                    <td style={{ ...CELL, textAlign: "center" }}>{mid}</td>
                    <td style={{ ...CELL, textAlign: "center" }}>{high}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="site-prose" style={{ marginTop: 20 }}>
            <ul>
              {page.median_rate_micros_per_hour !== null && (
                <li className="site-prose-li site-prose-li-disc">
                  Billed at a median {formatHourlyRate(page.median_rate_micros_per_hour / 1_000_000)}/hr, per second of run time
                  ({modes}).
                </li>
              )}
              {page.median_gpu_utilization !== null && (
                <li className="site-prose-li site-prose-li-disc">
                  The GPU was busy {Math.round(page.median_gpu_utilization)}% of the time in the median run, from
                  readings every 30 seconds ({page.utilization_runs} of {page.runs} runs reported them).
                </li>
              )}
            </ul>

            <h2 className="site-prose-h2">How this page is built</h2>
            <ul>
              <li className="site-prose-li site-prose-li-disc">
                Only finished runs whose owners chose to share them, plus operator runs on the live fleet, each
                labelled. Nothing is estimated or made up.
              </li>
              <li className="site-prose-li site-prose-li-disc">
                Only runs that started from the {page.template_label} template, ran at least a minute, ended
                normally, and whose charges match the wallet debits exactly.
              </li>
              <li className="site-prose-li site-prose-li-disc">
                The page shows only while {page.gpu_model} capacity is available to rent, and nothing on it
                identifies who ran a job, the job, the host or the image.
              </li>
            </ul>
          </div>

          <div className="site-hero-actions" style={{ marginTop: 32 }}>
            <Link href="/gpu-availability" className="site-button site-button-primary">
              See {page.gpu_model} availability <ArrowRight className="site-button-icon" />
            </Link>
            <Link href="/pricing" className="site-button site-button-ghost">
              Pricing
            </Link>
          </div>
        </article>
      </section>
    </div>
  );
}
