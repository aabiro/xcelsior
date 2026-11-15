"use client";

import Link from "next/link";
import { Monitor, Server } from "lucide-react";
import { useWorkspaceMode } from "@/lib/use-workspace-mode";
import { StartHostingButton } from "./start-hosting-button";
import { cn } from "@/lib/utils";

export function WorkspaceSwitch({ collapsed = false }: { collapsed?: boolean }) {
  const { mode, enabled, selectMode } = useWorkspaceMode();
  if (!enabled) return <div className="px-3 py-3"><StartHostingButton compact={collapsed} className={collapsed ? "w-full px-2" : "w-full"} /></div>;
  return <nav aria-label="Rent or host" className={cn("mx-3 my-3 flex gap-1 rounded-xl border border-border bg-navy/40 p-1", collapsed && "flex-col")}>
    {([
      { id: "rent", title: "Rent", href: "/dashboard/instances", Icon: Monitor },
      { id: "host", title: "Host", href: "/dashboard/hosting", Icon: Server },
    ] as const).map(({ id, title, href, Icon }) => <Link key={id} href={href} onClick={() => selectMode(id)} aria-current={mode === id ? "true" : undefined}
      title={collapsed ? title : undefined} aria-label={collapsed ? title : undefined}
      className={cn("flex min-w-0 flex-1 items-center justify-center gap-2 rounded-lg px-2 py-2 text-sm font-medium transition-colors", mode === id ? "bg-accent-cyan/10 text-accent-cyan ring-1 ring-accent-cyan/20" : "text-text-secondary hover:bg-surface-hover")}>
      <Icon className="h-4 w-4 shrink-0" />{!collapsed && title}
    </Link>)}
  </nav>;
}
