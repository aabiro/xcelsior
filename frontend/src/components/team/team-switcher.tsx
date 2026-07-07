"use client";

import { useCallback, useEffect, useState } from "react";
import { Users, ChevronDown, User, Loader2 } from "lucide-react";
import { useAuth } from "@/lib/auth";
import { useLocale } from "@/lib/locale";
import { fetchMyTeams, type TeamInfo } from "@/lib/api";
import { applyActiveTeamSwitch, getTeamContext } from "@/lib/team-context";
import { cn } from "@/lib/utils";
import { toast } from "sonner";

interface TeamSwitcherProps {
  className?: string;
  compact?: boolean;
}

export function TeamSwitcher({ className, compact = false }: TeamSwitcherProps) {
  const { user, refreshUser } = useAuth();
  const { t } = useLocale();
  const team = getTeamContext(user);
  const [teams, setTeams] = useState<TeamInfo[]>([]);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [switching, setSwitching] = useState<string | null>(null);

  const loadTeams = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetchMyTeams();
      setTeams(res.teams || []);
    } catch {
      setTeams([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadTeams();
  }, [loadTeams, user?.team_id]);

  const handleSwitch = async (teamId: string | null) => {
    if (switching) return;
    const current = user?.team_id || null;
    if ((teamId || null) === (current || null)) {
      setOpen(false);
      return;
    }
    setSwitching(teamId ?? "__personal__");
    try {
      await applyActiveTeamSwitch(teamId, refreshUser);
      await loadTeams();
      setOpen(false);
      toast.success(
        teamId
          ? t("dash.team.switch_success", {
              name: teams.find((x) => x.team_id === teamId)?.name || teamId,
            })
          : t("dash.team.switch_personal_success"),
      );
    } catch (err) {
      toast.error(err instanceof Error ? err.message : t("dash.team.switch_failed"));
    } finally {
      setSwitching(null);
    }
  };

  if (loading && teams.length === 0) return null;
  if (teams.length === 0) return null;

  const label = team.isTeamMember
    ? (team.teamName || t("dash.team"))
    : t("dash.team.personal_workspace");

  return (
    <div className={cn("relative", className)}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className={cn(
          "flex items-center gap-2 rounded-xl border border-border/60 px-2.5 py-1.5 text-sm transition-colors hover:bg-surface-hover",
          compact && "max-w-[11rem]",
        )}
        aria-expanded={open}
        aria-haspopup="listbox"
      >
        <Users className="h-3.5 w-3.5 shrink-0 text-accent-cyan" />
        <span className="truncate text-text-secondary">{label}</span>
        {switching ? (
          <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-text-muted" />
        ) : (
          <ChevronDown className={cn("h-3.5 w-3.5 shrink-0 text-text-muted transition-transform", open && "rotate-180")} />
        )}
      </button>

      {open && (
        <>
          <button
            type="button"
            className="dashboard-site-dropdown-backdrop"
            aria-label="Close team switcher"
            onClick={() => setOpen(false)}
          />
          <div
            role="listbox"
            className="absolute right-0 top-full z-[200] mt-1 min-w-[12rem] overflow-hidden rounded-xl shadow-xl"
          >
            <button
              type="button"
              role="option"
              aria-selected={!user?.team_id}
              onClick={() => void handleSwitch(null)}
              disabled={!!switching}
              className={cn(
                "flex w-full items-center gap-2 px-3 py-2 text-left text-sm transition-colors hover:bg-surface-hover",
                !user?.team_id && "bg-accent-cyan/10 text-accent-cyan",
              )}
            >
              <User className="h-3.5 w-3.5 shrink-0" />
              <span>{t("dash.team.personal_workspace")}</span>
            </button>
            <div className="border-t border-border/60" />
            {teams.map((entry) => (
              <button
                key={entry.team_id}
                type="button"
                role="option"
                aria-selected={user?.team_id === entry.team_id}
                onClick={() => void handleSwitch(entry.team_id)}
                disabled={!!switching}
                className={cn(
                  "flex w-full items-center gap-2 px-3 py-2 text-left text-sm transition-colors hover:bg-surface-hover",
                  user?.team_id === entry.team_id && "bg-accent-cyan/10 text-accent-cyan",
                )}
              >
                <Users className="h-3.5 w-3.5 shrink-0" />
                <span className="truncate">{entry.name}</span>
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}