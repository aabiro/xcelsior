#!/usr/bin/env node
/**
 * Verify all 51 cli.py cmd_* handlers have smoke tests and respond to --help.
 * Usage: node scripts/audit_cli_coverage.mjs
 */
import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const CLI_PY = path.join(REPO, "cli.py");
const TEST_PY = path.join(REPO, "tests", "test_cli_commands_coverage.py");

const CMD_RE = /^def (cmd_\w+)\(/gm;

function loadCommands() {
  const text = fs.readFileSync(CLI_PY, "utf8");
  const handlers = [];
  for (const m of text.matchAll(CMD_RE)) handlers.push(m[1]);
  return handlers.sort();
}

function loadTestCorpus() {
  const parts = [];
  for (const name of ["test_cli_commands_coverage.py", "test_cli.py"]) {
    const p = path.join(REPO, "tests", name);
    if (fs.existsSync(p)) parts.push(fs.readFileSync(p, "utf8"));
  }
  return parts.join("\n");
}

function hasTestSignal(handler, corpus) {
  const short = handler.slice(4); // cmd_run -> run
  return (
    corpus.includes(`def test_${handler}`) ||
    corpus.includes(`cli.${handler}`) ||
    corpus.includes(`test_cmd_${short}`)
  );
}

/** Parser dest names that differ from cmd_* → kebab-case heuristic. */
const HELP_ARGV = {
  cmd_provider_info: ["provider", "--help"],
  cmd_config: ["config", "list", "--help"],
};

function helpArgv(handler) {
  if (HELP_ARGV[handler]) return HELP_ARGV[handler];
  const name = handler.slice(4).replace(/_/g, "-");
  return [name, "--help"];
}

function main() {
  const handlers = loadCommands();
  const corpus = loadTestCorpus();
  const python = process.env.PYTHON || path.join(REPO, "venv", "bin", "python3");
  const checks = [];

  for (const handler of handlers) {
    const tested = hasTestSignal(handler, corpus);
    const argv = helpArgv(handler);
    const proc = spawnSync(python, ["-m", "cli", ...argv], {
      cwd: REPO,
      encoding: "utf8",
      timeout: 15000,
    });
    const helpOk = proc.status === 0;
    checks.push({
      handler,
      cli: argv.slice(0, -1).join(" "),
      tested,
      helpOk,
      ok: tested && helpOk,
      detail: !helpOk ? (proc.stderr || proc.stdout || "").trim().slice(0, 120) : "ok",
    });
  }

  const passed = checks.filter((c) => c.ok).length;
  const out = {
    checkedAt: new Date().toISOString(),
    summary: { total: checks.length, passed, failed: checks.length - passed },
    untested: checks.filter((c) => !c.tested).map((c) => c.handler),
    helpFailed: checks.filter((c) => !c.helpOk).map((c) => c.handler),
    checks,
  };

  console.log(JSON.stringify(out, null, 2));
  if (passed < checks.length) process.exit(1);
}

main();