# Xcelsior project setup

User Project: **Xcelsior**
Repo: `aabiro/xcelsior`
Site: https://xcelsior.ca

## Board

Create at https://github.com/users/aabiro/projects and link this repo.

Fields:

- Status: Backlog, Ready, In progress, In review, Done
- Priority: P0, P1, P2, P3
- Area: scheduler, billing, mesh, web, sdk
- Size: S, M, L

Views:

- Board grouped by Status
- Table filtered to Priority P0 or P1
- Roadmap on a 7-day Sprint starting Monday

## Branch rules

Already on `main` via ruleset `xcelsior-core`:

- No force-push, no deletion, linear history
- Pull request required, Copilot review on push
- Admin bypass stays on so solo merges are not blocked
- Do not require status checks until CI is green on main

## First backlog

- Fair-share / priority queue for consumer GPU jobs
- Stripe usage billing webhook idempotency
- Host heartbeat and VRAM admission control
- Web terminal auth boundary
- Public SDK smoke test in CI
