# Security policy

## Reporting a vulnerability

If you believe you have found a security issue in Xcelsior, tell us before
telling anyone else.

- **Email:** security@xcelsior.ca
- **GitHub:** if this repository shows a **Report a vulnerability** button
  under its Security tab, you can use it to send a private report instead.

Please do not open a public issue, discussion or pull request for a security
problem.

We will acknowledge within two business days and keep you updated until it is
resolved. We will not pursue legal action against good-faith research that
avoids privacy violations, service degradation, and access to data that is not
yours.

## What to include

- What is affected: a URL or API route, or a component and its version (worker
  agent, setup wizard, MCP package, SDK, desktop app).
- What you did, step by step, and what happened.
- What an attacker could gain, and what they would need first (an account, a
  host, a particular scope).

A proof of concept helps but is not required.

## In scope

- The website and dashboard at `xcelsior.ca`, and the API it calls.
- The MCP connector at `mcp.xcelsior.ca` and the `@xcelsior-gpu/mcp` package.
- The SSH gateway and the browser terminal for instances.
- The worker agent that providers install on their hosts, its installer
  (`install.sh`), and the `@xcelsior-gpu/wizard` setup wizard.
- The SDK, the CLI and the desktop app.
- The code in this repository.

Isolation between tenants, between a tenant and the host that runs it, and
between providers is the area we most want reports about.

## Ground rules for testing

- Test only with accounts and hosts you own. Never read, change or delete
  another user's data, instances, volumes or host.
- Do not launch workloads on other providers' hosts to probe them, and do not
  attack a provider's machine from an instance you rented on it.
- No denial-of-service or load testing, no spam, and no social engineering of
  staff, providers or customers.
- Stop and report as soon as you have shown the problem. Do not keep access you
  were not meant to have.

## Supported versions

- **Hosted service** (website, API, MCP connector, SSH gateway): only the
  running version. Fixes ship to it directly.
- **Worker agent:** the current release. Agents update through the platform's
  rollout, so a fix reaches hosts without a manual install.
- **npm packages** (`@xcelsior-gpu/mcp`, `@xcelsior-gpu/wizard`,
  `@xcelsior-gpu/sdk`): the latest published version.
