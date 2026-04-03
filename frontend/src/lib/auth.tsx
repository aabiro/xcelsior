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
import { getMe, logout as apiLogout, refreshToken } from "@/lib/api";

interface User {
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
}

/** Minutes of inactivity before showing the warning banner. */
const IDLE_WARN_MIN = 25;
/** Minutes of inactivity before we stop refreshing (session dies). */
const IDLE_LOGOUT_MIN = 30;

interface AuthState {
  user: User | null;
  loading: boolean;
  /** True when the user has been idle long enough to show the warning. */
  sessionExpiring: boolean;
  /** Call after login/register — fetches user profile via cookie. */
  login: () => Promise<void>;
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
  logout: async () => {},
  continueSession: () => {},
});

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [sessionExpiring, setSessionExpiring] = useState(false);

  // Track last user activity timestamp (ms)
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
      // Inactivity exceeded — let session die
      setUser(null);
      if (typeof window !== "undefined") {
        window.location.href = "/login?reason=idle";
      }
    }, IDLE_LOGOUT_MIN * 60_000);
  }, []);

  // Listen for user activity (only when logged in)
  useEffect(() => {
    if (!user) return;
    const EVENTS = ["mousedown", "keydown", "touchstart", "scroll"] as const;
    // Throttle: only reset if >30 s since last reset to avoid perf issues
    const handler = () => {
      if (Date.now() - lastActivity.current > 30_000) resetIdleTimers();
    };
    resetIdleTimers(); // start timers on login
    EVENTS.forEach((e) => window.addEventListener(e, handler, { passive: true }));
    return () => {
      EVENTS.forEach((e) => window.removeEventListener(e, handler));
      if (warnTimer.current) clearTimeout(warnTimer.current);
      if (logoutTimer.current) clearTimeout(logoutTimer.current);
    };
  }, [user, resetIdleTimers]);

  const login = useCallback(async () => {
    try {
      const res = await getMe();
      setUser(res.user);
    } catch {
      setUser(null);
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

  // Check session on mount — cookie is sent automatically
  useEffect(() => {
    getMe()
      .then((res) => setUser(res.user))
      .catch(() => setUser(null))
      .finally(() => setLoading(false));
  }, []);

  // Periodic session keepalive — refresh token every 10 minutes
  useEffect(() => {
    if (!user) return;
    const id = setInterval(() => {
      // Only refresh if user has been active recently
      if (Date.now() - lastActivity.current < IDLE_LOGOUT_MIN * 60_000) {
        refreshToken().catch(() => {
          setUser(null);
        });
      }
    }, 10 * 60 * 1000);
    return () => clearInterval(id);
  }, [user]);

  const value = useMemo(
    () => ({ user, loading, sessionExpiring, login, logout, continueSession }),
    [user, loading, sessionExpiring, login, logout, continueSession],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  return useContext(AuthContext);
}
