import type { MetadataRoute } from "next";
import { getAllPosts } from "@/lib/blog";
import { fetchReceiptPages } from "@/lib/receipt-pages";

/** Parse YYYY-MM-DD as local noon UTC to avoid timezone lastmod drift (F-021). */
function postLastModified(isoDate: string): Date {
  const [y, m, d] = isoDate.split("-").map((n) => parseInt(n, 10));
  if (!y || !m || !d) return new Date(isoDate);
  return new Date(Date.UTC(y, m - 1, d, 12, 0, 0));
}

const BASE_URL = "https://xcelsior.ca";

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const staticRoutes: MetadataRoute.Sitemap = [
    { url: BASE_URL, lastModified: new Date(), changeFrequency: "weekly", priority: 1.0 },
    { url: `${BASE_URL}/features`, lastModified: new Date(), changeFrequency: "monthly", priority: 0.8 },
    { url: `${BASE_URL}/pricing`, lastModified: new Date(), changeFrequency: "weekly", priority: 0.9 },
    { url: `${BASE_URL}/about`, lastModified: new Date(), changeFrequency: "monthly", priority: 0.6 },
    { url: `${BASE_URL}/privacy`, lastModified: new Date(), changeFrequency: "yearly", priority: 0.3 },
    { url: `${BASE_URL}/terms`, lastModified: new Date(), changeFrequency: "yearly", priority: 0.3 },
    { url: `${BASE_URL}/support`, lastModified: new Date(), changeFrequency: "monthly", priority: 0.5 },
    { url: `${BASE_URL}/download`, lastModified: new Date(), changeFrequency: "monthly", priority: 0.7 },
    { url: `${BASE_URL}/gpu-availability`, lastModified: new Date(), changeFrequency: "daily", priority: 0.85 },
    { url: `${BASE_URL}/mcp`, lastModified: new Date(), changeFrequency: "weekly", priority: 0.88 },
    { url: `${BASE_URL}/blog`, lastModified: new Date(), changeFrequency: "weekly", priority: 0.7 },
    { url: `${BASE_URL}/login`, changeFrequency: "yearly", priority: 0.4 },
    { url: `${BASE_URL}/register`, changeFrequency: "yearly", priority: 0.5 },
  ];

  const blogPosts: MetadataRoute.Sitemap = getAllPosts().map((post) => ({
    url: `${BASE_URL}/blog/${post.slug}`,
    lastModified: postLastModified(post.date),
    changeFrequency: "monthly" as const,
    priority: 0.6,
  }));

  // Workload pages exist only while they qualify (three shared runs, live
  // supply), so they are listed from the API, and the index only when it has any.
  const receipts = await fetchReceiptPages();
  const receiptPages: MetadataRoute.Sitemap =
    receipts && receipts.pages.length > 0
      ? [
          { url: `${BASE_URL}/receipts`, lastModified: new Date(), changeFrequency: "weekly", priority: 0.7 },
          ...receipts.pages.map((page) => ({
            url: `${BASE_URL}/receipts/${page.slug}`,
            lastModified: new Date(),
            changeFrequency: "weekly" as const,
            priority: 0.65,
          })),
        ]
      : [];

  return [...staticRoutes, ...blogPosts, ...receiptPages];
}
