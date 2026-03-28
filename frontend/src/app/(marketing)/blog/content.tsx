"use client";

import Link from "next/link";
import { ArrowRight, Calendar, Tag } from "lucide-react";
import { useLocale } from "@/lib/locale";
import type { BlogPost } from "@/lib/blog";

export function BlogContent({ posts }: { posts: BlogPost[] }) {
  const { t, locale } = useLocale();

  return (
    <div className="mx-auto max-w-4xl px-6 py-24">
      <div className="text-center mb-16">
        <h1 className="text-4xl font-bold md:text-5xl">{t("blog.title")}</h1>
        <p className="mt-4 text-lg text-text-secondary max-w-2xl mx-auto">
          {t("blog.subtitle")}
        </p>
      </div>

      {posts.length === 0 ? (
        <p className="text-center text-text-muted">{t("blog.empty")}</p>
      ) : (
        <div className="space-y-8">
          {posts.map((post) => (
            <article
              key={post.slug}
              className="group rounded-xl border border-border bg-surface p-6 card-hover"
            >
              <Link href={`/blog/${post.slug}`} className="block">
                <div className="flex items-center gap-3 text-xs text-text-muted mb-3">
                  <span className="flex items-center gap-1">
                    <Calendar className="h-3 w-3" />
                    {new Date(post.date).toLocaleDateString(locale === "fr" ? "fr-CA" : "en-CA", {
                      year: "numeric",
                      month: "long",
                      day: "numeric",
                    })}
                  </span>
                  <span>&middot;</span>
                  <span>{post.author}</span>
                </div>

                <h2 className="text-xl font-bold mb-2 group-hover:text-ice-blue transition-colors">
                  {post.title}
                </h2>
                <p className="text-sm text-text-secondary leading-relaxed mb-4">
                  {post.description}
                </p>

                <div className="flex items-center justify-between">
                  <div className="flex flex-wrap gap-2">
                    {post.tags.slice(0, 3).map((tag) => (
                      <span
                        key={tag}
                        className="inline-flex items-center gap-1 rounded-full bg-navy-light px-2.5 py-0.5 text-xs text-text-muted"
                      >
                        <Tag className="h-2.5 w-2.5" />
                        {tag}
                      </span>
                    ))}
                  </div>
                  <span className="text-sm text-ice-blue flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    {t("blog.read_more")} <ArrowRight className="h-3.5 w-3.5" />
                  </span>
                </div>
              </Link>
            </article>
          ))}
        </div>
      )}
    </div>
  );
}
