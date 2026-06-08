import { AuthProviderShell } from "@/components/AuthProviderShell";
import { privatePageMetadata } from "@/lib/page-metadata";

export const metadata = privatePageMetadata(
  "Create Account",
  "/register",
  "Create your Xcelsior account for Canada-first GPU compute.",
);

export default function RegisterLayout({ children }: { children: React.ReactNode }) {
  return <AuthProviderShell>{children}</AuthProviderShell>;
}