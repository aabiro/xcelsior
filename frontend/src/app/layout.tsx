import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Script from "next/script";
import { Providers } from "./providers";
import { RootClientWidgets } from "@/components/RootClientWidgets";
import { DeferredClientToaster } from "@/components/DeferredClientToaster";
import { BRAND_ASSET_ORIGIN, SITE_ASSETS } from "@/lib/brand-assets";
import "./globals.css";
import "@/components/marketing/marketing-theme.css";

const GA_ID = /^G-[A-Z0-9]+$/.test(process.env.NEXT_PUBLIC_GA_MEASUREMENT_ID ?? "")
  ? process.env.NEXT_PUBLIC_GA_MEASUREMENT_ID!
  : null;

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"], display: "swap", preload: false });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"], display: "swap", preload: false });

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
    default: "Xcelsior — The Agent-Native GPU Cloud",
    template: "%s | Xcelsior",
  },
  description:
    "Rent verified GPUs by the hour in CAD. Native MCP support for AI agents, real-time NVML telemetry, and dynamic spot pricing from $0.30 CAD/hr.",
  keywords: [
    "GPU compute",
    "MCP",
    "AI agents",
    "agent compute",
    "AI compute",
    "machine learning",
    "cloud GPU",
    "spot GPU",
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
        url: SITE_ASSETS.ogImage1200x630,
        width: 1200,
        height: 630,
        alt: "Xcelsior — The Agent-Native GPU Cloud",
      },
    ],
    title: "Xcelsior — The Agent-Native GPU Cloud",
    description:
      "Rent verified GPUs by the hour in CAD. Native MCP support for AI agents, real-time NVML telemetry, and dynamic spot pricing from $0.30 CAD/hr.",
    url: "https://xcelsior.ca",
    siteName: "Xcelsior",
    locale: "en_CA",
    alternateLocale: ["fr_CA"],
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    images: [SITE_ASSETS.ogImage1200x630],
    title: "Xcelsior — The Agent-Native GPU Cloud",
    description: "Fast GPUs for your AI agents. Native MCP, real telemetry, spot pricing. Ever upward.",
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
    icon: [
      { url: SITE_ASSETS.favicon16, sizes: "16x16", type: "image/png" },
      { url: SITE_ASSETS.favicon32, sizes: "32x32", type: "image/png" },
      { url: SITE_ASSETS.favicon48, sizes: "48x48", type: "image/png" },
    ],
    apple: [
      { url: SITE_ASSETS.appleTouchIcon180, sizes: "180x180", type: "image/png" },
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
    <html lang="en" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(){try{var t=localStorage.getItem('xcelsior-theme');if(t==='light'||t==='dark'){document.documentElement.classList.add(t);document.documentElement.dataset.theme=t;document.documentElement.style.colorScheme=t;}else{document.documentElement.classList.add('dark');document.documentElement.dataset.theme='dark';document.documentElement.style.colorScheme='dark';}}catch(e){document.documentElement.classList.add('dark');document.documentElement.dataset.theme='dark';}})();`,
          }}
        />
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
              logo: `${BRAND_ASSET_ORIGIN}${SITE_ASSETS.appGradientRounded512}`,
              description:
                "Rent verified GPUs by the hour in CAD. Native MCP support for AI agents, real-time NVML telemetry, and dynamic spot pricing from $0.30 CAD/hr.",
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
        className={`${geistSans.variable} ${geistMono.variable} font-sans antialiased`}
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
