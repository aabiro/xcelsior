import { privatePageMetadata } from "@/lib/page-metadata";

export const metadata = privatePageMetadata("Verify Email", "/verify-email");

export default function VerifyEmailLayout({ children }: { children: React.ReactNode }) {
  return children;
}