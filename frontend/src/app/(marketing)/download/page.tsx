import type { Metadata } from "next";
import { DownloadContent } from "./content";

export const metadata: Metadata = {
  title: "Download",
  description:
    "Get the Xcelsior desktop app for macOS, Windows, and Linux. Tray notifications, auto-updates, deep links, and offline routing. Mobile works right from the browser.",
  alternates: { canonical: "https://xcelsior.ca/download" },
  openGraph: {
    title: "Download | Xcelsior",
    description:
      "Desktop app with tray, auto-updates, and deep links. Mobile runs in the browser.",
    url: "https://xcelsior.ca/download",
  },
};

export default function DownloadPage() {
  return <DownloadContent />;
}
