"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { usePathname } from "next/navigation";
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
  const menuRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const pathname = usePathname();

  useEffect(() => { setOpen(false); }, [pathname]);
  useEffect(() => {
    if (!open) return;
    const dismiss = (event: PointerEvent) => {
      if (!menuRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpen(false);
        triggerRef.current?.focus();
      }
      if ((event.key === "ArrowDown" || event.key === "ArrowUp") && menuRef.current?.contains(event.target as Node)) {
        event.preventDefault();
        const options = Array.from(menuRef.current.querySelectorAll<HTMLButtonElement>('[role="option"]'));
        const current = options.indexOf(document.activeElement as HTMLButtonElement);
        options[(current + (event.key === "ArrowDown" ? 1 : options.length - 1)) % options.length]?.focus();
      }
    };
    document.addEventListener("pointerdown", dismiss);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", dismiss);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

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
    <div ref={menuRef} className={cn("relative", className)}>
      <button
        ref={triggerRef}
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
          <div
            role="listbox"
            aria-label="Workspace"
            className="dashboard-site-team-dropdown absolute right-0 top-full z-[200] mt-1 min-w-[12rem] overflow-hidden rounded-xl border border-[var(--edge)] bg-[var(--popover-solid)] shadow-[var(--shadow-panel)]"
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
