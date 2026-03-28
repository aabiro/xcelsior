import { ImageResponse } from "next/og";
import { getAllPosts, getPostBySlug } from "@/lib/blog";

export const alt = "Xcelsior Blog";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export function generateStaticParams() {
  return getAllPosts().map((post) => ({ slug: post.slug }));
}

export default async function BlogOG({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const post = getPostBySlug(slug);

  const title = post?.title ?? "Blog";
  const author = post?.author ?? "Xcelsior Team";
  const date = post?.date
    ? new Date(post.date).toLocaleDateString("en-CA", {
        year: "numeric",
        month: "long",
        day: "numeric",
      })
    : "";
  const tags = post?.tags?.slice(0, 3) ?? [];

  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          padding: "60px 80px",
          background: "linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%)",
          fontFamily: "sans-serif",
        }}
      >
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            height: 6,
            background: "linear-gradient(90deg, #dc2626, #f59e0b, #dc2626)",
          }}
        />

        {/* Logo + Blog label */}
        <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 32 }}>
          <div
            style={{
              width: 48,
              height: 48,
              borderRadius: 10,
              background: "#dc2626",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 28,
              fontWeight: 800,
              color: "white",
            }}
          >
            X
          </div>
          <span style={{ fontSize: 24, fontWeight: 600, color: "#64748b" }}>
            Xcelsior Blog
          </span>
        </div>

        {/* Title */}
        <p
          style={{
            fontSize: 48,
            fontWeight: 800,
            color: "#f8fafc",
            lineHeight: 1.2,
            marginTop: 0,
            marginBottom: 24,
            maxWidth: 900,
          }}
        >
          {title}
        </p>

        {/* Tags */}
        {tags.length > 0 && (
          <div style={{ display: "flex", gap: 12, marginBottom: 24 }}>
            {tags.map((tag) => (
              <span
                key={tag}
                style={{
                  padding: "6px 16px",
                  borderRadius: 999,
                  border: "1px solid #334155",
                  color: "#f59e0b",
                  fontSize: 16,
                  fontWeight: 600,
                }}
              >
                {tag}
              </span>
            ))}
          </div>
        )}

        {/* Author & date */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 16,
            position: "absolute",
            bottom: 48,
            left: 80,
          }}
        >
          <span style={{ fontSize: 18, color: "#94a3b8" }}>{author}</span>
          {date && (
            <>
              <span style={{ color: "#475569" }}>•</span>
              <span style={{ fontSize: 18, color: "#94a3b8" }}>{date}</span>
            </>
          )}
        </div>

        <span
          style={{
            position: "absolute",
            bottom: 48,
            right: 80,
            fontSize: 18,
            color: "#64748b",
          }}
        >
          xcelsior.ca/blog
        </span>
      </div>
    ),
    { ...size }
  );
}
