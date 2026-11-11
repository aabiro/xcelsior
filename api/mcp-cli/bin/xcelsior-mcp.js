#!/usr/bin/env node
// `npx @xcelsior-gpu/mcp`          the stdio server, with XCELSIOR_ACCESS_TOKEN
// `npx @xcelsior-gpu/mcp connect`  browser sign-in to the hosted server, no token
if (process.argv[2] === "connect") {
  const { connect } = await import("./connect.js");
  connect();
} else {
  await import("../dist/stdio.js");
}
