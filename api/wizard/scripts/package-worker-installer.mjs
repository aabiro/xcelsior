import { copyFileSync, mkdirSync, writeFileSync } from "node:fs";

// Ship the same signed-module installer used by the documented shell path.
// The wizard must not maintain a second, incomplete list of worker modules.
mkdirSync(new URL("../dist/assets/", import.meta.url), { recursive: true });
copyFileSync(new URL("../../scripts/install.sh", import.meta.url), new URL("../dist/assets/worker-install.sh", import.meta.url));
// Keep generated test fixtures and the demo out of the published tarball,
// including stale outputs left by incremental TypeScript builds.
writeFileSync(
  new URL("../dist/.npmignore", import.meta.url),
  ["src/__mocks__/", "src/**/__tests__/", "src/animation-demo.*", "src/**/*.test.*", ""].join("\n"),
);
