import { notFound } from "next/navigation";
import Link from "next/link";
import { getAllPosts, getPostBySlug } from "@/lib/blog";
import { ArrowLeft, Calendar, Tag } from "lucide-react";
import type { Metadata } from "next";

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
  return {
    title: post.title,
    description: post.description,
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
    publisher: {
      "@type": "Organization",
      name: "Xcelsior Computing Inc.",
      url: "https://xcelsior.ca",
      logo: { "@type": "ImageObject", url: "https://xcelsior.ca/xcelsior_icon_512x512.png" },
    },
    mainEntityOfPage: `https://xcelsior.ca/blog/${slug}`,
    keywords: post.tags.join(", "),
    image: post.image ? `https://xcelsior.ca${post.image}` : `https://xcelsior.ca/blog/${slug}/opengraph-image`,
  };

  return (
    <div className="mx-auto max-w-3xl px-6 py-24">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />
      <Link
        href="/blog"
        className="inline-flex items-center gap-1.5 text-sm text-text-muted hover:text-text-primary transition-colors mb-8"
      >
        <ArrowLeft className="h-3.5 w-3.5" /> Back to Blog
      </Link>

      <article>
        <header className="mb-10">
          <div className="flex items-center gap-3 text-sm text-text-muted mb-4">
            <span className="flex items-center gap-1">
              <Calendar className="h-3.5 w-3.5" />
              {new Date(post.date).toLocaleDateString("en-CA", {
                year: "numeric",
                month: "long",
                day: "numeric",
              })}
            </span>
            <span>&middot;</span>
            <span>{post.author}</span>
          </div>
          <h1 className="text-3xl font-bold md:text-4xl leading-tight mb-4">
            {post.title}
          </h1>
          <p className="text-lg text-text-secondary">{post.description}</p>
          <div className="flex flex-wrap gap-2 mt-4">
            {post.tags.map((tag) => (
              <span
                key={tag}
                className="inline-flex items-center gap-1 rounded-full bg-navy-light px-3 py-1 text-xs text-text-muted"
              >
                <Tag className="h-2.5 w-2.5" />
                {tag}
              </span>
            ))}
          </div>
        </header>

        {/* Render markdown content as styled prose */}
        <div className="prose-blog space-y-4 text-text-secondary leading-relaxed">
          {post.content.split("\n").map((line, i) => {
            const trimmed = line.trim();
            if (!trimmed) return null;

            // Headings
            if (trimmed.startsWith("## "))
              return (
                <h2 key={i} className="text-xl font-bold text-text-primary mt-8 mb-3">
                  {trimmed.slice(3)}
                </h2>
              );
            if (trimmed.startsWith("### "))
              return (
                <h3 key={i} className="text-lg font-semibold text-text-primary mt-6 mb-2">
                  {trimmed.slice(4)}
                </h3>
              );

            // Horizontal rules
            if (trimmed === "---") return <hr key={i} className="border-border my-8" />;

            // List items
            if (trimmed.startsWith("- "))
              return (
                <li key={i} className="ml-5 list-disc text-sm">
                  {renderInline(trimmed.slice(2))}
                </li>
              );
            if (/^\d+\.\s/.test(trimmed)) {
              const text = trimmed.replace(/^\d+\.\s/, "");
              return (
                <li key={i} className="ml-5 list-decimal text-sm">
                  {renderInline(text)}
                </li>
              );
            }

            // Table rows
            if (trimmed.startsWith("|")) {
              if (trimmed.replace(/[|\-\s]/g, "") === "") return null; // separator row
              const cells = trimmed
                .split("|")
                .filter(Boolean)
                .map((c) => c.trim());
              return (
                <div key={i} className="flex gap-4 text-sm font-mono border-b border-border/50 py-1.5">
                  {cells.map((cell, j) => (
                    <span key={j} className={`flex-1 ${j === 0 ? "font-medium text-text-primary" : ""}`}>
                      {renderInline(cell)}
                    </span>
                  ))}
                </div>
              );
            }

            // Paragraph
            return (
              <p key={i} className="text-sm leading-relaxed">
                {renderInline(trimmed)}
              </p>
            );
          })}
        </div>
      </article>

      {/* Back to blog */}
      <div className="mt-16 pt-8 border-t border-border">
        <Link
          href="/blog"
          className="inline-flex items-center gap-1.5 text-sm text-ice-blue hover:underline"
        >
          <ArrowLeft className="h-3.5 w-3.5" /> All posts
        </Link>
      </div>
    </div>
  );
}

/** Minimal inline markdown: **bold**, `code`, [links](href), *italic* */
function renderInline(text: string): React.ReactNode {
  const parts: React.ReactNode[] = [];
  // Combined regex for bold, code, links, italic
  const re = /(\*\*(.+?)\*\*)|(`(.+?)`)|(\[(.+?)\]\((.+?)\))|(\*(.+?)\*)/g;
  let last = 0;
  let match: RegExpExecArray | null;

  while ((match = re.exec(text)) !== null) {
    if (match.index > last) parts.push(text.slice(last, match.index));

    if (match[1]) {
      // **bold**
      parts.push(<strong key={match.index} className="text-text-primary font-medium">{match[2]}</strong>);
    } else if (match[3]) {
      // `code`
      parts.push(
        <code key={match.index} className="text-xs bg-navy-light px-1.5 py-0.5 rounded font-mono text-ice-blue">
          {match[4]}
        </code>,
      );
    } else if (match[5]) {
      // [text](href)
      const href = match[7] || "#";
      const isExternal = href.startsWith("http");
      parts.push(
        isExternal ? (
          <a key={match.index} href={href} target="_blank" rel="noopener noreferrer" className="text-ice-blue hover:underline">
            {match[6]}
          </a>
        ) : (
          <Link key={match.index} href={href} className="text-ice-blue hover:underline">
            {match[6]}
          </Link>
        ),
      );
    } else if (match[8]) {
      // *italic*
      parts.push(<em key={match.index}>{match[9]}</em>);
    }

    last = match.index + match[0].length;
  }

  if (last < text.length) parts.push(text.slice(last));
  return parts.length === 1 ? parts[0] : parts;
}
