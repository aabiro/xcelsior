import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { Toaster } from "sonner";
import { Providers } from "./providers";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"], display: "swap" });
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
  title: {
    default: "Xcelsior — Sovereign GPU Compute for Canada",
    template: "%s | Xcelsior",
  },
  description:
    "Canada-first GPU compute marketplace with data sovereignty, compliance automation, and competitive pricing. Ever upward.",
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
  alternates: { canonical: "https://xcelsior.ca" },
  openGraph: {
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "Xcelsior — Sovereign GPU Compute for Canada",
      },
    ],
    title: "Xcelsior — Sovereign GPU Compute for Canada",
    description:
      "Canada-first GPU compute marketplace. Data sovereignty, compliance automation, competitive pricing.",
    url: "https://xcelsior.ca",
    siteName: "Xcelsior",
    locale: "en_CA",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    images: ["/og-image.png"],
    title: "Xcelsior — Sovereign GPU Compute for Canada",
    description: "Canada-first GPU compute marketplace. Ever upward.",
  },
  robots: { index: true, follow: true },
  icons: {
    icon: "/favicon.svg",
    apple: [
      { url: "/xcelsior_icon_180x180.png", sizes: "180x180", type: "image/png" },
    ],
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <head>
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org",
              "@type": "Organization",
              name: "Xcelsior Computing Inc.",
              url: "https://xcelsior.ca",
              logo: "https://xcelsior.ca/xcelsior_icon_512x512.png",
              description:
                "Canadian-owned GPU compute marketplace with data sovereignty, compliance automation, and CAD pricing.",
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
        className={`${geistSans.variable} ${geistMono.variable} font-sans antialiased bg-navy text-text-primary`}
      >
        <Providers>
          {children}
        </Providers>
        <Toaster
          position="bottom-right"
          theme="dark"
          toastOptions={{
            style: {
              background: "#1e293b",
              border: "1px solid #334155",
              color: "#f8fafc",
            },
          }}
        />
      </body>
    </html>
  );
}
