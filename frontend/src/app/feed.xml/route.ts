import { getAllPosts } from "@/lib/blog";

const BASE_URL = "https://xcelsior.ca";

function postPubDate(isoDate: string): string {
  const [y, m, d] = isoDate.split("-").map((n) => parseInt(n, 10));
  if (!y || !m || !d) return new Date(isoDate).toUTCString();
  return new Date(Date.UTC(y, m - 1, d, 12, 0, 0)).toUTCString();
}

function escapeXml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

export function GET() {
  const posts = getAllPosts();

  const items = posts
    .map(
      (post) => `    <item>
      <title>${escapeXml(post.title)}</title>
      <link>${BASE_URL}/blog/${post.slug}</link>
      <guid isPermaLink="true">${BASE_URL}/blog/${post.slug}</guid>
      <description>${escapeXml(post.description)}</description>
      <pubDate>${postPubDate(post.date)}</pubDate>
      <author>hello@xcelsior.ca (${escapeXml(post.author)})</author>${
        post.tags.length
          ? post.tags.map((t) => `\n      <category>${escapeXml(t)}</category>`).join("")
          : ""
      }
    </item>`,
    )
    .join("\n");

  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>Xcelsior Blog</title>
    <link>${BASE_URL}/blog</link>
    <description>News, guides, and updates from Xcelsior — Canada-Grounded AI Compute — Cheapest Compliant Compute in Canada.</description>
    <language>en-ca</language>
    <lastBuildDate>${new Date().toUTCString()}</lastBuildDate>
    <atom:link href="${BASE_URL}/feed.xml" rel="self" type="application/rss+xml" />
${items}
  </channel>
</rss>`;

  return new Response(xml, {
    headers: {
      "Content-Type": "application/rss+xml; charset=utf-8",
      "Cache-Control": "s-maxage=3600, stale-while-revalidate",
    },
  });
}
