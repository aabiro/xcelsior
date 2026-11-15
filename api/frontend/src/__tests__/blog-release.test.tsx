// @vitest-environment node

import { readFile } from "node:fs/promises";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { getAllPosts, getPostBySlug } from "@/lib/blog";
import { generateMetadata } from "@/app/(marketing)/blog/[slug]/page";
import BlogOG from "@/app/(marketing)/blog/[slug]/opengraph-image";
import { GET as getBlogImage } from "@/app/(marketing)/blog/[slug]/image/route";

describe("Slurm and MCP blog release", () => {
  it("every explicitly authored blog image exists", async () => {
    for (const post of getAllPosts()) {
      if (post.image) {
        const image = await readFile(path.join(process.cwd(), "public", post.image));
        expect(image.byteLength, post.slug).toBeGreaterThan(0);
      }
    }
  });

  for (const slug of ["slurm-at-xcelsior", "mcp-server-release"]) {
    it(`${slug} has a canonical URL and sharing artwork`, async () => {
      const post = getPostBySlug(slug);
      expect(post).not.toBeNull();
      const imageUrl = post!.image ?? `/blog/${slug}/image`;
      const metadata = await generateMetadata({ params: Promise.resolve({ slug }) });
      expect(metadata.alternates?.canonical).toBe(`https://xcelsior.ca/blog/${slug}`);
      expect(metadata.openGraph).toMatchObject({
        type: "article",
        url: `https://xcelsior.ca/blog/${slug}`,
        images: [{ url: imageUrl, width: 1200, height: 630, alt: post!.title }],
      });
      expect(metadata.twitter).toMatchObject({
        card: "summary_large_image",
        images: [imageUrl],
      });

      const response = await BlogOG({ params: Promise.resolve({ slug }) });
      expect(response.headers.get("Content-Type")).toBe("image/png");
      const actual = Buffer.from(await response.arrayBuffer());
      if (post!.image) {
        const expected = await readFile(path.join(process.cwd(), "public", post!.image));
        expect(actual).toEqual(expected);
      } else {
        expect(actual.subarray(0, 8)).toEqual(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]));
      }
    });
  }

  it("generates MCP sharing artwork from the configured title", () => {
    const post = getPostBySlug("mcp-server-release");
    expect(post?.image).toBeUndefined();
    expect(post?.title).not.toContain("{mcpToolCount}");
  });

  it("orders posts on the same day by slug consistently", () => {
    const sameDate = getAllPosts().filter((post) => post.date === "2026-11-11");
    const slugs = sameDate.map((post) => post.slug);
    expect(slugs).toEqual([...slugs].sort((a, b) => a.localeCompare(b)));
    expect(slugs).toEqual(expect.arrayContaining(["slurm-at-xcelsior", "mcp-server-release"]));
  });

  it("posts without authored artwork advertise the stable generated-image URL", async () => {
    const slug = "aws-comparison";
    const metadata = await generateMetadata({ params: Promise.resolve({ slug }) });
    expect(metadata.twitter).toMatchObject({ images: [`/blog/${slug}/image`] });
  });

  it("does not generate an image for an unknown post", async () => {
    const response = await getBlogImage(new Request("https://xcelsior.ca/blog/missing/image"), {
      params: Promise.resolve({ slug: "missing" }),
    });
    expect(response.status).toBe(404);
  });
});
