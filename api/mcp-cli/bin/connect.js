// `npx @xcelsior-gpu/mcp connect`: sign in with the browser and connect to the
// hosted server (https://mcp.xcelsior.ca/mcp) through mcp-remote, with no token
// to copy. mcp-remote does the OAuth and the proxying; this only runs it and
// adds Xcelsior's sign-off to its log once the connection is up.
//
// stdin and stdout are the MCP channel and pass straight through. Only stderr,
// where mcp-remote logs, is read, line by line, and copied unchanged.

import { spawn } from "node:child_process";
import { pathToFileURL } from "node:url";

export const DEFAULT_URL = "https://mcp.xcelsior.ca/mcp";
// The line mcp-remote prints once the proxy is up. A minor range, so the text
// this waits for stays the same; if it ever changes, the sign-off is simply
// not printed and nothing else is affected.
export const MCP_REMOTE = "mcp-remote@^0.16.6";
export const READY_LINE = "Press Ctrl+C to exit";
export const SIGN_OFF = "👋 Thanks from Xcelsior. Your GPUs are standing by.";

/** Copy log text through, adding the sign-off after the ready line, once. */
export function createLogRelay(write) {
  let buffered = "";
  let signedOff = false;
  const emit = (line) => {
    write(line + "\n");
    if (!signedOff && line.includes(READY_LINE)) {
      signedOff = true;
      write(SIGN_OFF + "\n");
    }
  };
  return {
    push(chunk) {
      buffered += chunk;
      let newline;
      while ((newline = buffered.indexOf("\n")) !== -1) {
        emit(buffered.slice(0, newline).replace(/\r$/, ""));
        buffered = buffered.slice(newline + 1);
      }
    },
    end() {
      if (buffered) emit(buffered);
      buffered = "";
    },
  };
}

export function connect(argv = process.argv.slice(3), env = process.env) {
  const url = argv.find((arg) => /^https?:\/\//.test(arg)) || env.XCELSIOR_MCP_URL || DEFAULT_URL;
  // Tests point this at a stand-in; in use it is always npx.
  const command = env.XCELSIOR_MCP_REMOTE_COMMAND
    ? env.XCELSIOR_MCP_REMOTE_COMMAND.split(" ")
    : [process.platform === "win32" ? "npx.cmd" : "npx", "-y", MCP_REMOTE];
  const child = spawn(command[0], [...command.slice(1), url], {
    stdio: ["inherit", "inherit", "pipe"],
    shell: process.platform === "win32",
  });
  const relay = createLogRelay((text) => process.stderr.write(text));
  child.stderr.setEncoding("utf8");
  child.stderr.on("data", (chunk) => relay.push(chunk));
  child.stderr.on("end", () => relay.end());
  for (const signal of ["SIGINT", "SIGTERM"]) {
    process.on(signal, () => child.kill(signal));
  }
  child.on("error", (err) => {
    process.stderr.write(
      `[xcelsior-mcp] connect needs npx (Node.js 18 or later): ${err.message}\n`,
    );
    process.exitCode = 1;
  });
  child.on("exit", (code, signal) => {
    process.exitCode = code ?? (signal ? 1 : 0);
  });
  return child;
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  connect(process.argv.slice(2));
}
