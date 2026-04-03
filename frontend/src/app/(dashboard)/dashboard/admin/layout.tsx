"use client";

import { useAuth } from "@/lib/auth";
import { useRouter, usePathname } from "next/navigation";
import { useEffect } from "react";
import Link from "next/link";
import {
  LayoutDashboard, Users, DollarSign, Server, Activity,
  ShieldCheck, Bell, Shield,
} from "lucide-react";
import { cn } from "@/lib/utils";

const adminNav = [
  { href: "/dashboard/admin", label: "Overview", icon: LayoutDashboard, exact: true },
  { href: "/dashboard/admin/users", label: "Users", icon: Users },
  { href: "/dashboard/admin/revenue", label: "Revenue", icon: DollarSign },
  { href: "/dashboard/admin/infrastructure", label: "Infrastructure", icon: Server },
  { href: "/dashboard/admin/activity", label: "Activity", icon: Activity },
];

export default function AdminLayout({ children }: { children: React.ReactNode }) {
  const { user } = useAuth();
  const router = useRouter();
  const pathname = usePathname();
  const isAdmin = !!user?.is_admin || user?.role === "admin";

  useEffect(() => {
    if (user && !isAdmin) router.replace("/dashboard");
  }, [user, isAdmin, router]);

  if (!user || !isAdmin) return null;

  return (
    <div className="space-y-4">
      {/* Sub-navigation */}
      <nav className="flex gap-1 rounded-lg bg-surface p-1 overflow-x-auto">
        {adminNav.map((item) => {
          const active = item.exact
            ? pathname === item.href
            : pathname.startsWith(item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium whitespace-nowrap transition-all",
                active
                  ? "bg-card text-text-primary shadow-sm"
                  : "text-text-muted hover:text-text-primary hover:bg-surface-hover",
              )}
            >
              <item.icon className={cn("h-3.5 w-3.5 transition-colors", active && "text-accent-cyan drop-shadow-[0_0_4px_rgba(0,212,255,0.5)]")} />
              {item.label}
            </Link>
          );
        })}
      </nav>
      {children}
    </div>
  );
}
