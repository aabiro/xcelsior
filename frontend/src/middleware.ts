import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

const AUTH_COOKIE = "xcelsior_session";

const CSP_HEADER =
  "default-src 'self'; " +
  "script-src 'self' https://www.googletagmanager.com https://www.google-analytics.com https://js.stripe.com https://static.cloudflareinsights.com 'unsafe-inline'; " +
  "style-src 'self' 'unsafe-inline'; " +
  "img-src 'self' data: https:; " +
  "font-src 'self' data:; " +
  "connect-src 'self' https://www.google-analytics.com wss://xcelsior.ca https://api.web3modal.org https://*.walletconnect.org wss://relay.walletconnect.org https://pulse.walletconnect.org https://api.stripe.com; " +
  "frame-src 'self' https://js.stripe.com https://verify.walletconnect.org; " +
  "frame-ancestors 'self';";

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;
  const hasSession = request.cookies.has(AUTH_COOKIE);

  // Protect dashboard routes — redirect to login if no session cookie
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
