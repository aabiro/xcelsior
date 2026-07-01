"use client";

import Link from "next/link";
import { ArrowRight, Calendar, Tag } from "lucide-react";
import { useLocale } from "@/lib/locale";
import { formatBlogDate } from "@/lib/format-date";
import type { BlogPost } from "@/lib/blog";

export function BlogContent({ posts }: { posts: BlogPost[] }) {
  const { t, displayLocale } = useLocale();

  return (
    <>
      <section className="site-hero">
        <div className="site-grid-bg" aria-hidden />
        <div className="site-container">
          <div className="site-rails site-hero-rails" style={{ gridTemplateColumns: "1fr" }}>
            <div style={{ animation: "heroUp .7s ease both" }}>
              <div className="site-pill">
                <span className="site-live-dot" />
                <span>{t("blog.title")}</span>
              </div>
              <h1 className="site-hero-title">{t("blog.title")}</h1>
              <p className="site-hero-copy" style={{ maxWidth: 760 }}>{t("blog.subtitle")}</p>
            </div>
          </div>
        </div>
      </section>

      <div className="site-container">
        <section className="site-rails site-section">
          {posts.length === 0 ? (
            <div className="site-callout">
              <p className="site-callout-copy">{t("blog.empty")}</p>
            </div>
          ) : (
            <div className="site-blog-list">
              {posts.map((post) => (
                <article key={post.slug} className="site-blog-card">
                  <Link href={`/blog/${post.slug}`} className="site-blog-link">
                    <div className="site-blog-meta">
                      <span className="site-blog-meta-item">
                        <Calendar className="site-meta-icon" />
                        {formatBlogDate(post.date, displayLocale)}
                      </span>
                      <span>&middot;</span>
                      <span>{post.author}</span>
                    </div>
                    <h2 className="site-card-title" style={{ fontSize: 30, marginBottom: 14 }}>{post.title}</h2>
                    <p className="site-card-copy" style={{ fontSize: 16 }}>{post.description}</p>
                    <div className="site-blog-footer">
                      <div className="site-blog-tags">
                        {post.tags.slice(0, 3).map((tag) => (
                          <span key={tag} className="site-blog-tag">
                            <Tag className="site-tag-icon" />
                            {tag}
                          </span>
                        ))}
                      </div>
                      <span className="site-blog-readmore">
                        {t("blog.read_more")} <ArrowRight className="site-meta-icon" />
                      </span>
                    </div>
                  </Link>
                </article>
              ))}
            </div>
          )}
        </section>
      </div>
    </>
  );
}
