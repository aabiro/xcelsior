/**
 * Dashboard sidebar state kept in localStorage for the length of a session:
 * which nav groups are open, and each account's Rent/Host choice.
 *
 * Kept apart from the nav component so the login page can reset it without
 * loading the nav.
 */
export const NAV_GROUPS_KEY = "xcelsior-nav-groups-open";
export const WORKSPACE_MODE_KEY_PREFIX = "xcelsior-workspace-mode:";

/** Sign-out: the next session starts with every nav group closed and in Rent. */
export function resetSidebarState(): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.removeItem(NAV_GROUPS_KEY);
    for (let i = window.localStorage.length - 1; i >= 0; i--) {
      const key = window.localStorage.key(i);
      if (key?.startsWith(WORKSPACE_MODE_KEY_PREFIX)) window.localStorage.removeItem(key);
    }
  } catch {
    /* storage unavailable: nothing was remembered either */
  }
}
