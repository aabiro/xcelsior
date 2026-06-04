"use client";

import { usePathname } from "next/navigation";
import { Navbar } from "@/components/marketing/navbar";
import { Footer } from "@/components/marketing/footer";
import { MarketingChatWidget } from "@/components/marketing/MarketingChatWidget";

const NO_CHAT_PATHS = new Set(["/privacy", "/terms"]);

export function MarketingShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const showChat = !pathname || !NO_CHAT_PATHS.has(pathname);

  return (
    <>
      <Navbar />
      <main className="min-h-screen">{children}</main>
      <Footer />
      {showChat ? <MarketingChatWidget /> : null}
    </>
  );
}