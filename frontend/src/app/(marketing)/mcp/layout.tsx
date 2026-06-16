import { MarketingShell } from "@/components/marketing/MarketingShell";

export default function McpLayout({ children }: { children: React.ReactNode }) {
  return <MarketingShell>{children}</MarketingShell>;
}