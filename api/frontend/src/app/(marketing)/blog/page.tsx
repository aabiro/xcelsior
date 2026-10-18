import { getAllPosts } from "@/lib/blog";
import type { Metadata } from "next";
import dynamic from "next/dynamic";

const BlogContent = dynamic(
  () => import("./content").then((mod) => mod.BlogContent),
  { loading: () => <div className="min-h-[40vh]" aria-hidden /> },
);

export const metadata: Metadata = {
  title: "Blog",
  description:
    "News, guides, and insights from the Xcelsior team on GPU compute, Canadian AI policy, and platform updates for teams worldwide.",
  alternates: { canonical: "https://xcelsior.ca/blog" },
  openGraph: {
    title: "Blog | Xcelsior",
    description:
      "News, guides, and insights on GPU compute, Canadian AI policy, and platform updates for teams worldwide.",
    url: "https://xcelsior.ca/blog",
  },
  twitter: {
    title: "Blog | Xcelsior",
    description:
      "News, guides, and insights on GPU compute, Canadian AI policy, and platform updates for teams worldwide.",
  },
};

export default function BlogIndexPage() {
  const posts = getAllPosts();
  return <BlogContent posts={posts} />;
}
