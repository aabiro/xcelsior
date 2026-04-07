"use client";

import { useEffect, useState, useCallback, useMemo } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { StatCard } from "@/components/ui/stat-card";
import { Button } from "@/components/ui/button";
import { Badge, StatusBadge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { FadeIn, ScrollReveal, StaggerList, StaggerItem, CountUp, HoverCard } from "@/components/ui/motion";
import {
  Users, RefreshCw, Search, ChevronUp, ChevronDown,
  ArrowUpDown, Download, UserX, Wallet, BarChart3, UserCheck,
  MoreHorizontal, ShieldCheck, ShieldOff, ChevronRight, UserMinus,
} from "lucide-react";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import * as api from "@/lib/api";
import { useEventStream } from "@/hooks/useEventStream";
import { toast } from "sonner";

type User = Awaited<ReturnType<typeof api.fetchAdminUsers>>["users"][number];
type SortKey = "email" | "role" | "created_at" | "wallet_balance_cad" | "total_jobs";

export default function AdminUsersPage() {
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("created_at");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [page, setPage] = useState(0);
  const pageSize = 50;

  const load = useCallback(() => {
    setLoading(true);
    api.fetchAdminUsers()
      .then((r) => setUsers(Array.isArray(r.users) ? r.users : []))
      .catch(() => toast.error("Failed to load users"))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { load(); }, [load]);

  useEventStream({
    eventTypes: ["user_registered", "user_role_changed", "user_admin_toggled"],
    onEvent: load,
  });

  const [expandedUser, setExpandedUser] = useState<string | null>(null);
  const [actionMenu, setActionMenu] = useState<string | null>(null);
  const [teams, setTeams] = useState<api.AdminTeam[]>([]);
  const [removeFromTeam, setRemoveFromTeam] = useState<{ teamId: string; email: string; teamName: string } | null>(null);

  const loadTeams = useCallback(() => {
    api.fetchAdminTeams()
      .then((r) => setTeams(r.teams || []))
      .catch(() => { /* teams not available */ });
  }, []);

  useEffect(() => { loadTeams(); }, [loadTeams]);

  const teamsByUser = useMemo(() => {
    const map: Record<string, { team_id: string; name: string; role: string; owner_email: string }[]> = {};
    for (const t of teams) {
      for (const m of t.members) {
        if (!map[m.email]) map[m.email] = [];
        map[m.email].push({ team_id: t.team_id, name: t.name, role: m.role, owner_email: t.owner_email });
      }
    }
    return map;
  }, [teams]);

  const handleRemoveFromTeam = async () => {
    if (!removeFromTeam) return;
    try {
      await api.adminRemoveTeamMember(removeFromTeam.teamId, removeFromTeam.email);
      toast.success(`${removeFromTeam.email} removed from ${removeFromTeam.teamName}`);
      loadTeams();
      load();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to remove member");
    }
    setRemoveFromTeam(null);
  };

  const handleToggleAdmin = async (email: string) => {
    try {
      const res = await api.adminToggleAdmin(email);
      toast.success(`${email} admin ${res.is_admin ? "granted" : "revoked"}`);
      load();
    } catch { toast.error("Failed to toggle admin"); }
    setActionMenu(null);
  };

  const handleSetRole = async (email: string, role: string) => {
    try {
      await api.adminSetUserRole(email, role);
      toast.success(`${email} role set to ${role}`);
      load();
    } catch { toast.error("Failed to set role"); }
    setActionMenu(null);
  };

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else { setSortKey(key); setSortDir("desc"); }
    setPage(0);
  };

  const filtered = useMemo(() => {
    let list = users;
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      list = list.filter(
        (u) =>
          u.email?.toLowerCase().includes(q) ||
          u.role?.toLowerCase().includes(q) ||
          u.province?.toLowerCase().includes(q),
      );
    }
    list = [...list].sort((a, b) => {
      let av: string | number = (a as Record<string, unknown>)[sortKey] as string | number ?? "";
      let bv: string | number = (b as Record<string, unknown>)[sortKey] as string | number ?? "";
      if (typeof av === "string") av = av.toLowerCase();
      if (typeof bv === "string") bv = bv.toLowerCase();
      if (av < bv) return sortDir === "asc" ? -1 : 1;
      if (av > bv) return sortDir === "asc" ? 1 : -1;
      return 0;
    });
    return list;
  }, [users, searchQuery, sortKey, sortDir]);

  const paged = filtered.slice(page * pageSize, (page + 1) * pageSize);
  const totalPages = Math.ceil(filtered.length / pageSize);

  const totalWallet = useMemo(() => users.reduce((s, u) => s + (u.wallet_balance_cad ?? 0), 0), [users]);
  const activeCount = useMemo(() => users.filter((u) => u.is_active).length, [users]);
  const avgJobs = useMemo(() => users.length ? users.reduce((s, u) => s + (u.total_jobs ?? 0), 0) / users.length : 0, [users]);

  const exportCsv = () => {
    if (!filtered.length) { toast.error("No data to export"); return; }
    const keys: (keyof User)[] = ["email", "role", "is_admin", "wallet_balance_cad", "total_jobs", "province", "country", "created_at"];
    const csv = [keys.join(","), ...filtered.map((u) => keys.map((k) => `"${u[k] ?? ""}"`).join(","))].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = "xcelsior-users.csv"; a.click();
    URL.revokeObjectURL(url);
  };

  const SortHeader = ({ label, field }: { label: string; field: SortKey }) => (
    <th
      className="py-3 px-4 text-left font-medium cursor-pointer select-none hover:text-text-primary transition-colors"
      onClick={() => toggleSort(field)}
    >
      <span className="inline-flex items-center gap-1">
        {label}
        {sortKey === field ? (
          sortDir === "asc" ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />
        ) : (
          <ArrowUpDown className="h-3 w-3 opacity-30" />
        )}
      </span>
    </th>
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <h1 className="text-2xl font-bold">Users</h1>
        <div className="flex items-center gap-2">
          <span className="text-sm text-text-muted">{filtered.length} users</span>
          <Button variant="outline" size="sm" onClick={exportCsv}>
            <Download className="h-3.5 w-3.5" /> CSV
          </Button>
          <Button variant="outline" size="sm" onClick={load}>
            <RefreshCw className="h-3.5 w-3.5" /> Refresh
          </Button>
        </div>
      </div>

      {loading ? (
        <div className="space-y-4">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-4">
            {[...Array(4)].map((_, i) => <div key={i} className="h-28 rounded-xl bg-surface skeleton-pulse" />)}
          </div>
          <div className="h-10 rounded-lg bg-surface skeleton-pulse" />
          {[...Array(8)].map((_, i) => <div key={i} className="h-14 rounded-lg bg-surface skeleton-pulse" />)}
        </div>
      ) : users.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-20">
          <div className="flex h-20 w-20 items-center justify-center rounded-2xl bg-surface mb-6">
            <UserX className="h-10 w-10 text-text-muted" />
          </div>
          <h3 className="text-xl font-semibold mb-2">No users yet</h3>
          <p className="text-sm text-text-secondary max-w-md text-center">
            Users will appear here once they register on the platform.
          </p>
        </div>
      ) : (
        <>
          <StaggerList className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StaggerItem><StatCard label="Total Users" value={<CountUp value={users.length} />} icon={Users} glow="cyan" /></StaggerItem>
            <StaggerItem><StatCard label="Active (30d)" value={<CountUp value={activeCount} />} icon={UserCheck} glow="emerald" /></StaggerItem>
            <StaggerItem><StatCard label="Total Wallet" value={<CountUp value={totalWallet} prefix="$" />} icon={Wallet} glow="gold" /></StaggerItem>
            <StaggerItem><StatCard label="Avg Jobs/User" value={<CountUp value={Math.round(avgJobs * 10) / 10} />} icon={BarChart3} glow="violet" /></StaggerItem>
          </StaggerList>
        <ScrollReveal>
          <HoverCard>
          <Card className="glow-card brand-top-accent">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2"><Users className="h-4 w-4" /> All Users</CardTitle>
                <div className="relative w-72">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-text-muted" />
                  <Input
                    placeholder="Search by email, role, province..."
                    value={searchQuery}
                    onChange={(e) => { setSearchQuery(e.target.value); setPage(0); }}
                    className="pl-9 h-8 text-sm"
                  />
                </div>
              </div>
            </CardHeader>
            <CardContent>
              {paged.length === 0 ? (
                <p className="text-sm text-text-muted py-4">No users match your search</p>
              ) : (
                <>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-border text-text-secondary">
                          <th className="py-3 px-4 w-8"></th>
                          <SortHeader label="Email" field="email" />
                          <SortHeader label="Role" field="role" />
                          <th className="py-3 px-4 text-center font-medium">Access</th>
                          <th className="py-3 px-4 text-center font-medium">Status</th>
                          <SortHeader label="Wallet (CAD)" field="wallet_balance_cad" />
                          <SortHeader label="Jobs" field="total_jobs" />
                          <th className="py-3 px-4 text-left font-medium">Province</th>
                          <SortHeader label="Created" field="created_at" />
                          <th className="py-3 px-4 w-10"></th>
                        </tr>
                      </thead>
                      <tbody>
                        {paged.map((u) => (
                          <>
                          <tr
                            key={u.email}
                            className="border-b border-border/50 hover:bg-surface-hover hover:border-l-2 hover:border-l-accent-cyan transition-colors cursor-pointer"
                            onClick={() => setExpandedUser(expandedUser === u.email ? null : u.email)}
                          >
                            <td className="py-3 px-4">
                              <ChevronRight className={`h-3.5 w-3.5 text-text-muted transition-transform ${expandedUser === u.email ? "rotate-90" : ""}`} />
                            </td>
                            <td className="py-3 px-4 font-medium">{u.email}</td>
                            <td className="py-3 px-4">
                              <Badge variant="default">{u.role || "submitter"}</Badge>
                            </td>
                            <td className="py-3 px-4 text-center">
                              <Badge variant={u.is_admin ? "active" : "default"}>
                                {u.is_admin ? "Admin" : "Standard"}
                              </Badge>
                            </td>
                            <td className="py-3 px-4 text-center">
                              <StatusBadge status={u.is_active ? "active" : "offline"} />
                            </td>
                            <td className="py-3 px-4 text-right font-mono">
                              ${u.wallet_balance_cad?.toFixed(2) ?? "0.00"}
                            </td>
                            <td className="py-3 px-4 text-center font-mono">{u.total_jobs ?? 0}</td>
                            <td className="py-3 px-4 text-text-muted">{u.province || "—"}</td>
                            <td className="py-3 px-4 text-text-muted whitespace-nowrap">
                              {u.created_at ? new Date(u.created_at).toLocaleDateString() : "—"}
                            </td>
                            <td className="py-3 px-4 relative">
                              <button
                                onClick={(e) => { e.stopPropagation(); setActionMenu(actionMenu === u.email ? null : u.email); }}
                                className="p-1 rounded hover:bg-surface-hover transition-colors"
                              >
                                <MoreHorizontal className="h-4 w-4 text-text-muted" />
                              </button>
                              {actionMenu === u.email && (
                                <div className="absolute right-4 top-10 z-50 w-48 rounded-lg border border-border bg-card shadow-lg py-1 text-sm"
                                  onClick={(e) => e.stopPropagation()}>
                                  <button onClick={() => handleToggleAdmin(u.email)} className="flex w-full items-center gap-2 px-3 py-2 hover:bg-surface-hover transition-colors">
                                    {u.is_admin ? <ShieldOff className="h-3.5 w-3.5" /> : <ShieldCheck className="h-3.5 w-3.5" />}
                                    {u.is_admin ? "Revoke Admin" : "Grant Admin"}
                                  </button>
                                  <div className="border-t border-border my-1" />
                                  <p className="px-3 py-1 text-[10px] text-text-muted uppercase tracking-wide">Set Role</p>
                                  {["submitter", "provider"].map((role) => (
                                    <button key={role} onClick={() => handleSetRole(u.email, role)} className={`flex w-full items-center gap-2 px-3 py-1.5 hover:bg-surface-hover transition-colors capitalize ${(u.role || "submitter") === role ? "text-accent-cyan font-medium" : ""}`}>
                                      {role}
                                    </button>
                                  ))}
                                </div>
                              )}
                            </td>
                          </tr>
                          {expandedUser === u.email && (
                            <tr key={`${u.email}-detail`} className="bg-surface/50">
                              <td colSpan={10} className="px-8 py-4">
                                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm">
                                  <div>
                                    <p className="text-[10px] uppercase tracking-wide text-text-muted mb-1">Wallet</p>
                                    <p className="font-mono font-medium">${u.wallet_balance_cad?.toFixed(2) ?? "0.00"} CAD</p>
                                  </div>
                                  <div>
                                    <p className="text-[10px] uppercase tracking-wide text-text-muted mb-1">Total Jobs</p>
                                    <p className="font-mono font-medium">{u.total_jobs ?? 0}</p>
                                  </div>
                                  <div>
                                    <p className="text-[10px] uppercase tracking-wide text-text-muted mb-1">Location</p>
                                    <p>{u.province || "—"}{u.country ? `, ${u.country}` : ""}</p>
                                  </div>
                                  <div>
                                    <p className="text-[10px] uppercase tracking-wide text-text-muted mb-1">Created</p>
                                    <p>{u.created_at ? new Date(u.created_at).toLocaleString() : "—"}</p>
                                  </div>
                                </div>
                                {/* Team memberships */}
                                {teamsByUser[u.email] && teamsByUser[u.email].length > 0 && (
                                  <div className="mt-4 pt-3 border-t border-border/50">
                                    <p className="text-[10px] uppercase tracking-wide text-text-muted mb-2">Teams</p>
                                    <div className="flex flex-wrap gap-2">
                                      {teamsByUser[u.email].map((tm) => {
                                        const isOwner = tm.owner_email === u.email;
                                        return (
                                          <div key={tm.team_id} className="flex items-center gap-2 rounded-lg border border-border/60 bg-navy-light/30 px-3 py-1.5 text-sm">
                                            <Users className="h-3.5 w-3.5 text-text-muted" />
                                            <span>{tm.name}</span>
                                            <Badge variant="default">{tm.role}</Badge>
                                            {isOwner && <span className="text-[10px] text-accent-gold font-medium">OWNER</span>}
                                            {!isOwner && (
                                              <button
                                                onClick={(e) => { e.stopPropagation(); setRemoveFromTeam({ teamId: tm.team_id, email: u.email, teamName: tm.name }); }}
                                                className="ml-1 p-0.5 rounded text-text-muted hover:text-accent-red transition-colors"
                                                title={`Remove from ${tm.name}`}
                                              >
                                                <UserMinus className="h-3.5 w-3.5" />
                                              </button>
                                            )}
                                          </div>
                                        );
                                      })}
                                    </div>
                                  </div>
                                )}
                              </td>
                            </tr>
                          )}
                          </>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  {totalPages > 1 && (
                    <div className="flex items-center justify-between pt-4 border-t border-border mt-4">
                      <span className="text-xs text-text-muted">
                        Page {page + 1} of {totalPages}
                      </span>
                      <div className="flex gap-2">
                        <Button variant="outline" size="sm" disabled={page === 0} onClick={() => setPage((p) => p - 1)}>
                          Previous
                        </Button>
                        <Button variant="outline" size="sm" disabled={page >= totalPages - 1} onClick={() => setPage((p) => p + 1)}>
                          Next
                        </Button>
                      </div>
                    </div>
                  )}
                </>
              )}
            </CardContent>
          </Card>
          </HoverCard>
        </ScrollReveal>
        </>
      )}

      <ConfirmDialog
        open={removeFromTeam !== null}
        title="Remove Team Member"
        description={`Remove ${removeFromTeam?.email ?? ""} from team "${removeFromTeam?.teamName ?? ""}"? They will lose access to shared team resources.`}
        confirmLabel="Remove"
        cancelLabel="Cancel"
        variant="danger"
        onConfirm={handleRemoveFromTeam}
        onCancel={() => setRemoveFromTeam(null)}
      />
    </div>
  );
}
