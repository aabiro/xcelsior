import { privatePageMetadata } from "@/lib/page-metadata";

export const metadata = privatePageMetadata("Reset Password", "/reset-password");

export default function ResetPasswordLayout({ children }: { children: React.ReactNode }) {
  return children;
}