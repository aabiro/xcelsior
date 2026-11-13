import { getPostBySlug } from "@/lib/blog";
import BlogOG from "../opengraph-image";

export { generateStaticParams } from "../opengraph-image";
export const dynamic = "force-static";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ slug: string }> },
) {
  const { slug } = await params;
  if (!getPostBySlug(slug)) return new Response(null, { status: 404 });
  return BlogOG({ params: Promise.resolve({ slug }) });
}
