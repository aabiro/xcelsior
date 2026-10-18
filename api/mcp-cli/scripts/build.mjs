#!/usr/bin/env node
// Builds ../mcp, copies its output to dist/, then checks the dependencies.
//
// Under `npm publish` or `npm run`, npm 12 exports the user's ~/.npmrc as
// npm_config_* variables, and a nested `npm ci` reads them as flags: with
// allow-scripts set it refuses (EALLOWSCRIPTS), so the publish failed before
// building. The child commands get the environment without them, the same
// rule as the wizard's packageManagerEnv().

import { execFileSync } from "node:child_process";
import { cpSync, rmSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const server = resolve(root, "../mcp");
const env = Object.fromEntries(Object.entries(process.env).filter(([name]) =>
    !/^npm_(config|package|lifecycle|execpath|node_execpath|command)(_|$)/i.test(name)));
const npm = process.platform === "win32" ? "npm.cmd" : "npm";
const run = (cmd, args, cwd) => execFileSync(cmd, args, { cwd, env, stdio: "inherit", shell: process.platform === "win32" });

run(npm, ["ci"], server);
run(npm, ["run", "build"], server);
rmSync(resolve(root, "dist"), { recursive: true, force: true });
cpSync(resolve(server, "dist"), resolve(root, "dist"), { recursive: true });
run(process.execPath, [resolve(root, "scripts/check-runtime-deps.mjs")], root);
