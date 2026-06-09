import { defineConfig } from "vitest/config";

export default defineConfig({
    test: {
        environment: "node",
        globals: true,
        projects: [
            {
                test: {
                    name: "unit",
                    include: ["src/__tests__/**/*.test.{ts,tsx}"],
                    exclude: ["src/__tests__/api-client-hardening.test.ts"],
                    alias: {
                        "../checks.js": new URL("./src/__mocks__/checks.ts", import.meta.url).pathname,
                        "../api-client.js": new URL("./src/__mocks__/api-client.ts", import.meta.url).pathname,
                    },
                },
            },
            {
                test: {
                    name: "integration",
                    include: ["src/__tests__/api-client-hardening.test.ts"],
                },
            },
        ],
    },
});
