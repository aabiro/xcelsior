/** The one URL for an endpoint's page. `/dashboard/inference/<id>` only redirects here. */
export function endpointHref(endpointId: string): string {
  return `/dashboard/inference?endpoint=${encodeURIComponent(endpointId)}`;
}
