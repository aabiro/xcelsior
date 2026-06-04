/** Routes that should probe /api/auth/me on mount (dashboard redirect, auth flows). */
const SESSION_ON_MOUNT = [
  /^\/dashboard(\/|$)/,
  /^\/login$/,
  /^\/register$/,
  /^\/accept-invite$/,
  /^\/setup-2fa$/,
] as const;

export function needsSessionOnMount(pathname: string | null | undefined): boolean {
  if (!pathname) return false;
  return SESSION_ON_MOUNT.some((re) => re.test(pathname));
}