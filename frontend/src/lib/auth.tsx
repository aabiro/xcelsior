"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
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

interface AuthState {
  user: User | null;
  loading: boolean;
  /** Call after login/register — fetches user profile via cookie. */
  login: () => Promise<void>;
  /** POST /api/auth/logout, clear cookie + state. */
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthState>({
  user: null,
  loading: true,
  login: async () => {},
  logout: async () => {},
});

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

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
      refreshToken().catch(() => {
        // Refresh failed — session is dead, clear user to trigger redirect
        setUser(null);
      });
    }, 10 * 60 * 1000);
    return () => clearInterval(id);
  }, [user]);

  const value = useMemo(
    () => ({ user, loading, login, logout }),
    [user, loading, login, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  return useContext(AuthContext);
}
