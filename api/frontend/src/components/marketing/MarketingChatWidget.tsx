"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";

const ChatWidget = dynamic(
  () => import("@/components/ChatWidget").then((m) => ({ default: m.ChatWidget })),
  { ssr: false },
);

export function MarketingChatWidget() {
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (typeof window.requestIdleCallback === "function") {
      const id = window.requestIdleCallback(() => setReady(true), { timeout: 5000 });
      return () => window.cancelIdleCallback(id);
    }
    const id = window.setTimeout(() => setReady(true), 2500);
    return () => window.clearTimeout(id);
  }, []);

  if (!ready) return null;
  return <ChatWidget />;
}