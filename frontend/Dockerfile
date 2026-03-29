# ── Stage 1: Dependencies ────────────────────────────────────────────
FROM node:22-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci --ignore-scripts

# ── Stage 2: Build ───────────────────────────────────────────────────
FROM node:22-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

ARG NEXT_PUBLIC_API_URL=https://xcelsior.ca
ARG NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}
ENV NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=${NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY}
ENV NEXT_TELEMETRY_DISABLED=1
ENV DOCKER_BUILD=1

RUN npm run build

# ── Stage 3: Production ─────────────────────────────────────────────
FROM node:22-alpine AS runner

LABEL org.opencontainers.image.title="Xcelsior Frontend" \
      org.opencontainers.image.description="Next.js frontend for Xcelsior — Canadian sovereign GPU marketplace" \
      org.opencontainers.image.url="https://xcelsior.ca" \
      org.opencontainers.image.source="https://github.com/aabiro/xcelsior" \
      org.opencontainers.image.vendor="Xcelsior" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

# Copy standalone output
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public
# Copy blog content for dynamic reads
COPY --from=builder /app/content ./content

USER nextjs

EXPOSE 3000

ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget -qO- http://127.0.0.1:3000/ || exit 1

CMD ["node", "server.js"]
