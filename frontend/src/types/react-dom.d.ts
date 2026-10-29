declare module "react-dom" {
  import type { ReactNode, ReactPortal } from "react";

  export function createPortal(
    children: ReactNode,
    container: Element | DocumentFragment,
    key?: string | null,
  ): ReactPortal;
}
// Only what the hydration test needs: rendering a server pass and hydrating it.
declare module "react-dom/client" {
  import type { ReactNode } from "react";

  export interface Root {
    render(children: ReactNode): void;
    unmount(): void;
  }

  export function hydrateRoot(
    container: Element | Document,
    initialChildren: ReactNode,
    options?: { onRecoverableError?: (error: unknown, errorInfo: unknown) => void },
  ): Root;
}

declare module "react-dom/server" {
  import type { ReactNode } from "react";

  export function renderToString(element: ReactNode): string;
}
