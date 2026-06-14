"use client";

import dynamic from "next/dynamic";

const Toaster = dynamic(() => import("sonner").then((m) => ({ default: m.Toaster })), {
  ssr: false,
});

export function ClientToaster() {
  return (
    <Toaster
      position="bottom-right"
      theme="dark"
      closeButton
      visibleToasts={4}
      toastOptions={{
        duration: 4000,
        style: {
          background: "#1e293b",
          border: "1px solid #334155",
          color: "#f8fafc",
        },
        classNames: {
          closeButton: "xcelsior-toast-dismiss",
        },
      }}
    />
  );
}