"use client";

import type { ReactNode } from "react";
import { AppKitProvider } from "@reown/appkit/react";
import { BitcoinAdapter } from "@reown/appkit-adapter-bitcoin";
import { bitcoin } from "@reown/appkit/networks";

const projectId = process.env.NEXT_PUBLIC_REOWN_PROJECT_ID?.trim() || "";
const appUrl = process.env.NEXT_PUBLIC_APP_URL?.trim() || "https://xcelsior.ca";

const metadata = {
  name: "Xcelsior",
  description: "Canadian GPU compute marketplace with CAD wallet funding.",
  url: appUrl,
  icons: [`${appUrl}/favicon.svg`],
};

const bitcoinAdapter = projectId
  ? new BitcoinAdapter({
      projectId,
    })
  : null;

const networks: [typeof bitcoin] = [bitcoin];

export const isWalletConnectConfigured = Boolean(projectId && bitcoinAdapter);

export function WalletConnectProvider({
  children,
}: {
  children: ReactNode;
}) {
  if (!bitcoinAdapter || !projectId) {
    return <>{children}</>;
  }

  return (
    <AppKitProvider
      adapters={[bitcoinAdapter]}
      networks={networks}
      defaultNetwork={bitcoin}
      projectId={projectId}
      metadata={metadata}
      themeMode="dark"
      themeVariables={{
        "--apkt-accent": "#f59e0b",
        "--apkt-color-mix": "#060a13",
        "--apkt-color-mix-strength": 18,
        "--apkt-border-radius-master": "18px",
        "--apkt-font-family": "var(--font-geist-sans)",
        "--apkt-z-index": 90,
      }}
    >
      {children}
    </AppKitProvider>
  );
}
