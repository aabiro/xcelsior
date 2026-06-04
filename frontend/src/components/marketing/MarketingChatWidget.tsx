"use client";

import dynamic from "next/dynamic";

const ChatWidget = dynamic(
  () => import("@/components/ChatWidget").then((m) => ({ default: m.ChatWidget })),
  { ssr: false },
);

export function MarketingChatWidget() {
  return <ChatWidget />;
}