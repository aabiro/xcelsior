import { getAllPosts } from "@/lib/blog";
import type { Metadata } from "next";
import { BlogContent } from "./content";

export const metadata: Metadata = {
  title: "Blog",
  description:
    "News, guides, and insights from the Xcelsior team on sovereign GPU compute, Canadian AI policy, and platform updates.",
  alternates: { canonical: "https://xcelsior.ca/blog" },
  openGraph: {
    title: "Blog | Xcelsior",
    description:
      "News, guides, and insights on sovereign GPU compute, Canadian AI policy, and platform updates.",
    url: "https://xcelsior.ca/blog",
  },
  twitter: {
    title: "Blog | Xcelsior",
    description:
      "News, guides, and insights on sovereign GPU compute, Canadian AI policy, and platform updates.",
  },
};

export default function BlogIndexPage() {
  const posts = getAllPosts();
  return <BlogContent posts={posts} />;
}
