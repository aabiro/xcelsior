import { privatePageMetadata } from "@/lib/page-metadata";

export const metadata = privatePageMetadata("Set Up Two-Factor Authentication", "/setup-2fa");

export default function Setup2faLayout({ children }: { children: React.ReactNode }) {
  return children;
}