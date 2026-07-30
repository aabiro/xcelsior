import type { MetadataRoute } from "next";

export default function robots(): MetadataRoute.Robots {
  return {
    rules: [
      {
        userAgent: "*",
        allow: "/",
        // /site-assets/reference/ holds unlinked design-system dumps that are served
        // but are not product pages — keep them out of the index.
        disallow: ["/dashboard/", "/api/", "/site-assets/reference/", "/brand-system/"],
      },
    ],
    sitemap: "https://xcelsior.ca/sitemap.xml",
  };
}
