import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

const AUTH_COOKIE = "xcelsior_session";

const NO_STORE_PATHS = new Set([
  "/login",
  "/register",
  "/forgot-password",
  "/reset-password",
  "/setup-2fa",
  "/verify-email",
  "/accept-invite",
  "/delete-data",
  "/deactivate",
  "/~offline",
]);

const CSP_HEADER =
  "default-src 'self'; " +
  "script-src 'self' https://www.googletagmanager.com https://www.google-analytics.com https://js.stripe.com https://connect-js.stripe.com https://static.cloudflareinsights.com 'unsafe-inline'; " +
  "style-src 'self' 'unsafe-inline'; " +
  "img-src 'self' data: blob: https:; " +
  "font-src 'self' data:; " +
  "connect-src 'self' https://www.google-analytics.com https://www.googletagmanager.com https://static.cloudflareinsights.com wss://xcelsior.ca https://api.web3modal.org https://*.walletconnect.org wss://relay.walletconnect.org https://pulse.walletconnect.org https://api.stripe.com https://js.stripe.com https://connect-js.stripe.com https://us.i.posthog.com https://us-assets.i.posthog.com; " +
  "frame-src 'self' https://js.stripe.com https://connect-js.stripe.com https://verify.walletconnect.org; " +
  "frame-ancestors 'self';";

export async function proxy(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // Meta posts the data-deletion callback to the public page URL. The page
  // itself is the GET document. The POST is the signed callback.
  if (
    request.method === "POST"
    && (pathname === "/delete-data" || pathname === "/deactivate")
  ) {
    const backend = (
      process.env.INTERNAL_API_URL
      || process.env.NEXT_PUBLIC_API_URL
      || "https://xcelsior.ca"
    ).replace(/\/$/, "");
    const upstream = await fetch(`${backend}/api/auth/oauth/facebook/delete-data`, {
      method: "POST",
      headers: {
        "content-type": request.headers.get("content-type") || "application/x-www-form-urlencoded",
      },
      body: await request.text(),
      cache: "no-store",
    });
    return new NextResponse(await upstream.text(), {
      status: upstream.status,
      headers: {
        "content-type": upstream.headers.get("content-type") || "application/json",
        "cache-control": "private, no-store",
      },
    });
  }

  const hasSession = request.cookies.has(AUTH_COOKIE);

  // Protect dashboard routes, redirect to login if no session cookie
  if (pathname.startsWith("/dashboard")) {
    if (!hasSession) {
      const loginUrl = new URL("/login", request.url);
      loginUrl.searchParams.set("redirect", pathname);
      return NextResponse.redirect(loginUrl);
    }
    const response = NextResponse.next();
    response.headers.set("Content-Security-Policy", CSP_HEADER);
    return response;
  }

  // Allow auth pages to load (login page handles redirect if already authenticated)
  if (pathname === "/login" || pathname === "/register") {
    const response = NextResponse.next();
    response.headers.set("Content-Security-Policy", CSP_HEADER);
    return response;
  }

  const response = NextResponse.next();
  response.headers.set("Content-Security-Policy", CSP_HEADER);
  if (NO_STORE_PATHS.has(pathname) || pathname.startsWith("/dashboard")) {
    response.headers.set("Cache-Control", "private, no-store, max-age=0, must-revalidate");
  }
  return response;
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     */
    "/((?!_next/static|_next/image|favicon.ico).*)",
  ],
};
