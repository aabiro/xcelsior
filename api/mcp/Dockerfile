# The frontend's base image pin (api/frontend/Dockerfile); tests/test_release_facts.py
# fails if the two differ.
ARG NODE_IMAGE=node:22-alpine@sha256:1314fb30ec37592d5845e5a160d522c0e680972aa534549940b8d73bde88481a
FROM ${NODE_IMAGE} AS build
WORKDIR /app
# Phase 10 / §19.6: locked install only — never fall back to npm install.
COPY package.json package-lock.json ./
RUN npm ci
COPY tsconfig.json ./
COPY src ./src
RUN npm run build

FROM ${NODE_IMAGE}
WORKDIR /app
ENV NODE_ENV=production
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=3100
ENV MCP_PATH=/mcp
# Non-root runtime (Phase 10 privilege drop).
RUN addgroup -S xcelsior && adduser -S -G xcelsior xcelsior
COPY package.json package-lock.json ./
RUN npm ci --omit=dev && chown -R xcelsior:xcelsior /app
COPY --from=build --chown=xcelsior:xcelsior /app/dist ./dist
COPY --chown=xcelsior:xcelsior assets ./assets
USER xcelsior
EXPOSE 3100
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget -qO- http://127.0.0.1:3100/health || exit 1
CMD ["node", "dist/index.js"]
