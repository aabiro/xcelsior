"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { ReactNode } from "react";
import { usePathname } from "next/navigation";
import { getMe, logout as apiLogout, refreshToken } from "@/lib/api";
import { needsSessionOnMount } from "@/lib/session-routes";

/** Revoke the session server-side then redirect to the login page. */
function forceLogout(reason: string) {
  apiLogout()
    .catch(() => {})
    .finally(() => {
      if (typeof window !== "undefined") {
        window.location.href = `/login?reason=${encodeURIComponent(reason)}`;
      }
    });
}

export interface User {
  user_id: string;
  email: string;
  name?: string;
  role: string;
  is_admin?: boolean;
  country?: string;
  province?: string;
  avatar_url?: string;
  customer_id?: string;
  provider_id?: string;
  team_id?: string | null;
  team_name?: string;
  team_role?: string;
  team_plan?: string;
  billing_customer_id?: string;
  team_can_manage_billing?: boolean;
  team_can_write_instances?: boolean;
}

/** Minutes of inactivity before showing the warning banner. */
const IDLE_WARN_MIN = 55;
/** Minutes of inactivity before we stop refreshing (session dies). */
const IDLE_LOGOUT_MIN = 60;

interface AuthState {
  user: User | null;
  loading: boolean;
  /** True when the user has been idle long enough to show the warning. */
  sessionExpiring: boolean;
  /** Call after login/register — fetches user profile via cookie. */
  login: () => Promise<void>;
  /** Silently re-fetch /api/auth/me and update user in context. */
  refreshUser: () => Promise<void>;
  /** POST /api/auth/logout, clear cookie + state. */
  logout: () => Promise<void>;
  /** Reset idle timer & refresh token — called from the "Continue Session" banner. */
  continueSession: () => void;
}

const AuthContext = createContext<AuthState>({
  user: null,
  loading: true,
  sessionExpiring: false,
  login: async () => {},
  refreshUser: async () => {},
  logout: async () => {},
  continueSession: () => {},
});

export function AuthProvider({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(() => needsSessionOnMount(null));
  const [sessionExpiring, setSessionExpiring] = useState(false);
  const sessionFetched = useRef(false);

  // Track last user activity timestamp (ms)
  // useRef seeds this once on first render and ignores the value thereafter,
  // so Date.now() here is stable, not an unstable impure read.
  // eslint-disable-next-line react-hooks/purity
  const lastActivity = useRef(Date.now());
  const warnTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const logoutTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const resetIdleTimers = useCallback(() => {
    lastActivity.current = Date.now();
    setSessionExpiring(false);
    if (warnTimer.current) clearTimeout(warnTimer.current);
    if (logoutTimer.current) clearTimeout(logoutTimer.current);
    warnTimer.current = setTimeout(() => setSessionExpiring(true), IDLE_WARN_MIN * 60_000);
    logoutTimer.current = setTimeout(() => {
      // Inactivity exceeded — revoke session server-side and redirect
      setUser(null);
      forceLogout("idle");
    }, IDLE_LOGOUT_MIN * 60_000);
  }, []);

  // Listen for user activity (only when logged in)
  useEffect(() => {
    if (!user) return;
    const EVENTS = ["mousedown", "keydown", "touchstart", "scroll"] as const;
    // Throttle: only reset if >30 s since last reset.
    // Once the expiry banner is showing, ignore activity — user must
    // explicitly click "Continue Session" to stay logged in.
    const handler = () => {
      if (sessionExpiring) return;
      if (Date.now() - lastActivity.current > 30_000) resetIdleTimers();
    };
    if (!sessionExpiring) resetIdleTimers(); // start timers on login (but don't restart if banner is up)
    EVENTS.forEach((e) => window.addEventListener(e, handler, { passive: true }));
    return () => {
      EVENTS.forEach((e) => window.removeEventListener(e, handler));
      // Only clear the warn timer here — never clear logoutTimer during
      // cleanup because the sessionExpiring state change triggers a re-render
      // whose cleanup runs with the OLD closure (sessionExpiring=false),
      // which would cancel the pending 30-min logout before it fires.
      if (warnTimer.current) clearTimeout(warnTimer.current);
    };
  }, [user, sessionExpiring, resetIdleTimers]);

  const login = useCallback(async () => {
    try {
      const res = await getMe();
      setUser(res.user);
    } catch {
      setUser(null);
    }
  }, []);

  const refreshUser = useCallback(async () => {
    try {
      const res = await getMe();
      setUser(res.user);
    } catch {
      // Silently ignore — stale user stays in context
    }
  }, []);

  const logout = useCallback(async () => {
    try {
      await apiLogout();
    } catch {
      /* cookie already cleared or network error — fine */
    }
    setUser(null);
    if (typeof window !== "undefined") {
      window.location.href = "/login";
    }
  }, []);

  const continueSession = useCallback(() => {
    resetIdleTimers();
    refreshToken().catch(() => {});
  }, [resetIdleTimers]);

  // Probe session only on routes that need it (dashboard, login redirect, etc.)
  useEffect(() => {
    if (!needsSessionOnMount(pathname)) {
      if (!sessionFetched.current) {
        setLoading(false);
      }
      return;
    }
    let cancelled = false;
    setLoading(true);
    getMe()
      .then((res) => {
        if (!cancelled) setUser(res.user ?? null);
      })
      .catch(() => {
        if (!cancelled) setUser(null);
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
          sessionFetched.current = true;
        }
      });
    return () => {
      cancelled = true;
    };
  }, [pathname]);

  // Periodic session keepalive — refresh token every 10 minutes
  useEffect(() => {
    if (!user) return;
    const id = setInterval(() => {
      // Only refresh if user has been active recently
      if (Date.now() - lastActivity.current < IDLE_LOGOUT_MIN * 60_000) {
        // Silently ignore refresh failures — a network hiccup should not
        // force logout when the user is actively using the app.
        refreshToken().catch(() => {});
      }
    }, 10 * 60 * 1000);
    return () => clearInterval(id);
  }, [user]);

  const value = useMemo(
    () => ({ user, loading, sessionExpiring, login, refreshUser, logout, continueSession }),
    [user, loading, sessionExpiring, login, refreshUser, logout, continueSession],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  return useContext(AuthContext);
}
