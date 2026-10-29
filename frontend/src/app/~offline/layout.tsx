import { privatePageMetadata } from "@/lib/page-metadata";

export const metadata = privatePageMetadata("Offline", "/~offline", "You are offline. Cached Xcelsior pages may still be available.");

export default function OfflineLayout({ children }: { children: React.ReactNode }) {
  return children;
}