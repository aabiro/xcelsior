import { defineConfig } from "vitest/config";

export default defineConfig({
    test: {
        environment: "node",
        globals: true,
        setupFiles: ["./src/__tests__/setup.ts"],
        projects: [
            {
                test: {
                    name: "unit",
                    setupFiles: ["./src/__tests__/setup.ts"],
                    // Full-App Ink renders + async gate/service checks are slow
                    // under host load; the 5s default flakes. Give real headroom.
                    testTimeout: 20000,
                    hookTimeout: 30000,
                    include: ["src/__tests__/**/*.test.{ts,tsx}"],
                    exclude: ["src/__tests__/api-client-hardening.test.ts"],
                },
            },
            {
                test: {
                    name: "integration",
                    setupFiles: ["./src/__tests__/setup.ts"],
                    include: ["src/__tests__/api-client-hardening.test.ts"],
                },
            },
        ],
    },
});
