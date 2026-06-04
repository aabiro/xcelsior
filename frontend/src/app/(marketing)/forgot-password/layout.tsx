import { privatePageMetadata } from "@/lib/page-metadata";

export const metadata = privatePageMetadata("Forgot Password", "/forgot-password");

export default function ForgotPasswordLayout({ children }: { children: React.ReactNode }) {
  return children;
}