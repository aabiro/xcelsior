// The parent (marketing) layout already wraps pages in <MarketingShell>
// (navbar + footer). Re-wrapping here doubled the header and footer, so this
// layout is a passthrough.
export default function McpLayout({ children }: { children: React.ReactNode }) {
  return children;
}