import { redirect } from "next/navigation";
import { endpointHref } from "@/features/serverless/endpoint-href";

/**
 * An endpoint lives at `/dashboard/inference?endpoint=<id>`. This path is what
 * the Quick Deploy sheet and the endpoint sidebar used to link to, and it was a
 * 404 (prod, 2026-11-22). Old links and bookmarks land on the real page.
 */
export default async function EndpointPathRedirect({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  redirect(endpointHref(decodeURIComponent(id)));
}
