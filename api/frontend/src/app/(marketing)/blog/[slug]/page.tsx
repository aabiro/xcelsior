import { notFound } from "next/navigation";
import Link from "next/link";
import { ArrowLeft, Calendar, Tag } from "lucide-react";
import type { Metadata } from "next";
import { getAllPosts, getPostBySlug } from "@/lib/blog";
import { formatBlogDate } from "@/lib/format-date";

interface Props {
  params: Promise<{ slug: string }>;
}

export async function generateStaticParams() {
  return getAllPosts().map((post) => ({ slug: post.slug }));
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params;
  const post = getPostBySlug(slug);
  if (!post) return { title: "Post Not Found" };
  const description =
    post.description.length > 160 ? `${post.description.slice(0, 157)}...` : post.description;
  return {
    title: post.title,
    description,
    alternates: { canonical: `https://xcelsior.ca/blog/${slug}` },
    openGraph: {
      title: post.title,
      description: post.description,
      type: "article",
      publishedTime: post.date,
      authors: [post.author],
    },
  };
}

export default async function BlogPostPage({ params }: Props) {
  const { slug } = await params;
  const post = getPostBySlug(slug);
  if (!post) notFound();

  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "BlogPosting",
    headline: post.title,
    description: post.description,
    datePublished: post.date,
    author: { "@type": "Person", name: post.author },
    publisher: { "@id": "https://xcelsior.ca/#organization" },
    mainEntityOfPage: `https://xcelsior.ca/blog/${slug}`,
    keywords: post.tags.join(", "),
    image: post.image ? `https://xcelsior.ca${post.image}` : `https://xcelsior.ca/blog/${slug}/opengraph-image`,
  };

  return (
    <div className="site-container">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />
      <section className="site-rails site-section site-article-shell">
        <Link href="/blog" className="site-article-back">
          <ArrowLeft className="site-meta-icon" /> Back to Blog
        </Link>

        <article className="site-article">
          <header className="site-article-header">
            <div className="site-blog-meta">
              <span className="site-blog-meta-item">
                <Calendar className="site-meta-icon" />
                {formatBlogDate(post.date, "en")}
              </span>
              <span>&middot;</span>
              <span>{post.author}</span>
            </div>
            <h1 className="site-article-title">{post.title}</h1>
            <p className="site-article-description">{post.description}</p>
            <div className="site-blog-tags">
              {post.tags.map((tag) => (
                <span key={tag} className="site-blog-tag">
                  <Tag className="site-tag-icon" />
                  {tag}
                </span>
              ))}
            </div>
          </header>

          <div className="site-prose">
            {post.content.split("\n").map((line, i) => {
              const trimmed = line.trim();
              if (!trimmed) return null;

              if (trimmed.startsWith("## ")) {
                return (
                  <h2 key={i} className="site-prose-h2">
                    {trimmed.slice(3)}
                  </h2>
                );
              }
              if (trimmed.startsWith("### ")) {
                return (
                  <h3 key={i} className="site-prose-h3">
                    {trimmed.slice(4)}
                  </h3>
                );
              }

              if (trimmed === "---") return <hr key={i} className="site-prose-hr" />;

              if (trimmed.startsWith("- ")) {
                return (
                  <li key={i} className="site-prose-li site-prose-li-disc">
                    {renderInline(trimmed.slice(2))}
                  </li>
                );
              }
              if (/^\d+\.\s/.test(trimmed)) {
                const text = trimmed.replace(/^\d+\.\s/, "");
                return (
                  <li key={i} className="site-prose-li site-prose-li-decimal">
                    {renderInline(text)}
                  </li>
                );
              }

              if (trimmed.startsWith("|")) {
                if (trimmed.replace(/[|\-\s]/g, "") === "") return null;
                const cells = trimmed
                  .split("|")
                  .filter(Boolean)
                  .map((cell) => cell.trim());
                return (
                  <div key={i} className="site-prose-table-row">
                    {cells.map((cell, j) => (
                      <span key={j} className={j === 0 ? "site-prose-table-cell site-prose-table-cell-heading" : "site-prose-table-cell"}>
                        {renderInline(cell)}
                      </span>
                    ))}
                  </div>
                );
              }

              return (
                <p key={i} className="site-prose-p">
                  {renderInline(trimmed)}
                </p>
              );
            })}
          </div>
        </article>

        <div className="site-article-footer">
          <Link href="/blog" className="site-article-back site-article-back-accent">
            <ArrowLeft className="site-meta-icon" /> All posts
          </Link>
        </div>
      </section>
    </div>
  );
}

function renderInline(text: string): React.ReactNode {
  const parts: React.ReactNode[] = [];
  const re = /(\*\*(.+?)\*\*)|(`(.+?)`)|(\[(.+?)\]\((.+?)\))|(\*(.+?)\*)/g;
  let last = 0;
  let match: RegExpExecArray | null;

  while ((match = re.exec(text)) !== null) {
    if (match.index > last) parts.push(text.slice(last, match.index));

    if (match[1]) {
      parts.push(<strong key={match.index} className="site-prose-strong">{match[2]}</strong>);
    } else if (match[3]) {
      parts.push(
        <code key={match.index} className="site-inline-code">
          {match[4]}
        </code>,
      );
    } else if (match[5]) {
      const href = match[7] || "#";
      const isExternal = href.startsWith("http");
      parts.push(
        isExternal ? (
          <a key={match.index} href={href} rel="noopener noreferrer" className="site-inline-link">
            {match[6]}
          </a>
        ) : (
          <Link key={match.index} href={href} className="site-inline-link">
            {match[6]}
          </Link>
        ),
      );
    } else if (match[8]) {
      parts.push(<em key={match.index} className="site-prose-em">{match[9]}</em>);
    }

    last = match.index + match[0].length;
  }

  if (last < text.length) parts.push(text.slice(last));
  return parts.length === 1 ? parts[0] : parts;
}
