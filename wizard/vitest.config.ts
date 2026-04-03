import { defineConfig } from "vitest/config";

export default defineConfig({
    test: {
        environment: "node",
        globals: true,
        include: ["src/__tests__/**/*.test.{ts,tsx}"],
        alias: {
            // Route imports to mocks in test environment
            "../checks.js": new URL("./src/__mocks__/checks.ts", import.meta.url).pathname,
            "../api-client.js": new URL("./src/__mocks__/api-client.ts", import.meta.url).pathname,
        },
    },
});
