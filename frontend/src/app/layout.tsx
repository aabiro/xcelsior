import type { Metadata, Viewport } from "next";
import { Geist } from "next/font/google";
import Script from "next/script";
import { Providers } from "./providers";
import { RootClientWidgets } from "@/components/RootClientWidgets";
import { DeferredClientToaster } from "@/components/DeferredClientToaster";
import "./globals.css";

const GA_ID = /^G-[A-Z0-9]+$/.test(process.env.NEXT_PUBLIC_GA_MEASUREMENT_ID ?? "")
  ? process.env.NEXT_PUBLIC_GA_MEASUREMENT_ID!
  : null;

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"], display: "swap", preload: false });

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: [
    { media: "(prefers-color-scheme: dark)", color: "#060a13" },
    { media: "(prefers-color-scheme: light)", color: "#ffffff" },
  ],
};

export const metadata: Metadata = {
  applicationName: "Xcelsior",
  title: {
    default: "Xcelsior — The Cheapest Compliant GPU Compute in Canada",
    template: "%s | Xcelsior",
  },
  description:
    "Rent verified GPUs by the hour in Canadian dollars — sovereignty, PIPEDA, and clean hydro built in. From $0.30 CAD/hr, with dynamic spot pricing up to 70% off.",
  keywords: [
    "GPU compute",
    "Canada",
    "data sovereignty",
    "PIPEDA",
    "AI compute",
    "machine learning",
    "cloud GPU",
    "sovereign cloud",
  ],
  metadataBase: new URL("https://xcelsior.ca"),
  alternates: {
    canonical: "https://xcelsior.ca",
    languages: {
      "en-CA": "https://xcelsior.ca",
      "fr-CA": "https://xcelsior.ca",
      "x-default": "https://xcelsior.ca",
    },
    types: { "application/rss+xml": "https://xcelsior.ca/feed.xml" },
  },
  openGraph: {
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "The Cheapest Compliant GPU Compute in Canada",
      },
    ],
    title: "The Cheapest Compliant GPU Compute in Canada",
    description:
      "Rent verified GPUs by the hour in Canadian dollars — sovereignty, PIPEDA, and clean hydro built in. From $0.30 CAD/hr, with dynamic spot pricing up to 70% off.",
    url: "https://xcelsior.ca",
    siteName: "Xcelsior",
    locale: "en_CA",
    alternateLocale: ["fr_CA"],
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    images: ["/og-image.png"],
    title: "The Cheapest Compliant GPU Compute in Canada",
    description: "The Cheapest Compliant GPU Compute in Canada. Ever upward.",
  },
  appleWebApp: {
    capable: true,
    statusBarStyle: "default",
    title: "Xcelsior",
  },
  robots: { index: true, follow: true },
  verification: {
    google: process.env.NEXT_PUBLIC_GOOGLE_SITE_VERIFICATION || undefined,
  },
  icons: {
    icon: "/favicon.svg",
    apple: [
      { url: "/xcelsior_icon_180x180.png", sizes: "180x180", type: "image/png" },
    ],
  },
  manifest: "/manifest.webmanifest",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link rel="dns-prefetch" href="https://docs.xcelsior.ca" />
        {GA_ID && (
          <>
            <Script
              src={`https://www.googletagmanager.com/gtag/js?id=${GA_ID}`}
              strategy="lazyOnload"
            />
            <Script id="gtag-init" strategy="lazyOnload">
              {`window.dataLayer=window.dataLayer||[];function gtag(){dataLayer.push(arguments);}gtag('js',new Date());gtag('config','${GA_ID}',{anonymize_ip:true});`}
            </Script>
          </>
        )}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org",
              "@type": "Organization",
              "@id": "https://xcelsior.ca/#organization",
              name: "Xcelsior Compute Inc.",
              url: "https://xcelsior.ca",
              logo: "https://xcelsior.ca/xcelsior_icon_512x512.png",
              description:
                "Rent verified GPUs by the hour in Canadian dollars — sovereignty, PIPEDA, and clean hydro built in. From $0.30 CAD/hr, with dynamic spot pricing up to 70% off.",
              foundingDate: "2024",
              address: {
                "@type": "PostalAddress",
                addressCountry: "CA",
              },
              sameAs: [],
              contactPoint: {
                "@type": "ContactPoint",
                email: "hello@xcelsior.ca",
                contactType: "customer service",
              },
            }),
          }}
        />
      </head>
      <body
        suppressHydrationWarning
        className={`${geistSans.variable} font-sans antialiased bg-navy text-text-primary`}
      >
        <Providers>
          {children}
          <RootClientWidgets />
        </Providers>
        <DeferredClientToaster />
      </body>
    </html>
  );
}
