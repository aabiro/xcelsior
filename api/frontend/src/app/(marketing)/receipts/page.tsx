import Link from "next/link";
import { ArrowRight } from "lucide-react";
import type { Metadata } from "next";
import { connection } from "next/server";
import { formatDuration, formatMicros } from "@/lib/format-units";
import { fetchReceiptPages, monthLabel, pageTitle } from "@/lib/receipt-pages";

const DESCRIPTION =
  "What GPU workloads actually cost on Xcelsior: billed time and cost from real runs their owners shared, per quick-start template and GPU.";

export async function generateMetadata(): Promise<Metadata> {
  await connection();
  const data = await fetchReceiptPages();
  return {
    title: "Real run costs",
    description: DESCRIPTION,
    alternates: { canonical: "https://xcelsior.ca/receipts" },
    openGraph: { title: "Real run costs | Xcelsior", description: DESCRIPTION, url: "https://xcelsior.ca/receipts" },
    // An empty index is not worth a search result.
    robots: data && data.pages.length > 0 ? undefined : { index: false, follow: true },
  };
}

export default async function ReceiptPagesIndex() {
  // Rendered per request, so a failed read is never baked into the page (at
  // build time or during an API blip). The API answer itself is cached; see
  // fetchReceiptPages.
  await connection();
  const data = await fetchReceiptPages();
  return (
    <div className="site-container">
      <section className="site-rails site-section site-article-shell">
        <header className="site-article-header">
          <div className="site-eyebrow">Real runs</div>
          <h1 className="site-article-title">What workloads actually cost</h1>
          <p className="site-article-description">
            Each page is one quick-start template on one GPU, built from finished runs whose owners chose to
            share them, plus labelled operator runs on the live fleet. Every figure is what the ledger billed;
            nothing is estimated.
          </p>
        </header>

        {data === null ? (
          <div className="site-empty-state" role="status" style={{ marginTop: 32 }}>
            <p>These pages could not be loaded just now. Try again in a minute.</p>
          </div>
        ) : data.pages.length === 0 ? (
          <div className="site-empty-state" style={{ marginTop: 32 }} data-testid="receipt-pages-empty">
            <p>
              No pages yet. A page appears once {data.minRuns} real runs of the same template on the same GPU
              have been shared, and only while that GPU is available to rent.
            </p>
            <p>
              Finished a run from a quick-start template? You can share it from its instance page, under the
              receipt. Your name, the job, the host and the image are never shown.
            </p>
            <Link href="/gpu-availability" className="site-inline-link">
              See available GPUs
            </Link>
          </div>
        ) : (
          <div className="site-blog-list" style={{ marginTop: 32 }} data-testid="receipt-pages-list">
            {data.pages.map((page) => (
              <article key={page.slug} className="site-blog-card">
                <Link href={`/receipts/${page.slug}`} className="site-blog-link">
                  <h2 className="site-card-title">{pageTitle(page)}</h2>
                  <p className="site-card-copy">
                    Typical run {formatDuration(page.median_billed_seconds)} for{" "}
                    {formatMicros(page.median_total_micros, { exact: true })} · {page.runs} runs
                    {monthLabel(page.latest_month) ? ` · updated ${monthLabel(page.latest_month)}` : ""}
                  </p>
                  <span className="site-blog-readmore">
                    See the runs <ArrowRight className="site-meta-icon" />
                  </span>
                </Link>
              </article>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
