<!-- residency-guard: documents-removal — roadmap entries recording shipped-then-removed UI -->
# Xcelsior — UI/UX Roadmap & Comprehensive TODO

*Created: 2026-02-13 — Gap audit of 135 API endpoints vs dashboard UI*
*Updated: 2026-02-15 — Phase UI-1 through UI-14 partially implemented (v2.8.0)*
*Current state: ~165 endpoints now have UI exposure, auth backend + password reset + marketplace search + UX polish implemented*

---

## Competitor Reference (RunPod / Vast.ai / Lambda / CoreWeave)

Standard GPU cloud platform UX surfaces:

| Feature | RunPod | Vast.ai | Lambda | CoreWeave | Xcelsior |
|---------|--------|---------|--------|-----------|----------|
| OAuth login (Google/GitHub) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Email/password signup | ✅ | ✅ | ✅ | ✅ | ✅ |
| User profile & settings | ✅ | ✅ | ✅ | ✅ | ✅ |
| API key management | ✅ | ✅ | ✅ | ✅ | ✅ |
| SSH key management | ✅ | ✅ | ✅ | ✅ | ✅ |
| Credits/wallet/deposit | ✅ | ✅ | ✅ | ✅ | ✅ |
| Billing history/invoices | ✅ | ✅ | ✅ | ✅ | ✅ |
| Pricing page | ✅ | ✅ | ✅ | ✅ | ✅ |
| Team/org management | ✅ | ❌ | ✅ | ✅ | ❌ |
| Role-based access | ✅ 4 roles | ❌ | ✅ | ✅ | ✅ |
| Audit logs UI | ✅ | ❌ | ❌ | ✅ | ✅ |
| Instance detail view | ✅ | ✅ | ✅ | ✅ | ✅ |
| Reserved pricing purchase | ✅ | ✅ | ✅ | Sales | ✅ |
| Spot/preemptible bidding | ✅ | ✅ | ❌ | ✅ | Partial |
| Marketplace search/filter | N/A | ✅ | N/A | N/A | ✅ |
| Provider dashboard | N/A | ✅ | N/A | N/A | ✅ |
| Trust/reputation badges | N/A | ✅ DLPerf | N/A | N/A | ✅ |
| Storage/artifacts UI | ✅ | ✅ | ✅ | ✅ | ✅ |
| Usage analytics charts | ✅ | ✅ | ✅ | ✅ | ✅ |
| Notification preferences | ✅ | ✅ | ✅ | ✅ | Partial |

---

## Phase UI-1: Authentication & User Identity (CRITICAL)

The dashboard is currently unauthenticated. Every competitor requires login.

### UI-1.1 — Login/Signup Page
- [x] Create standalone login page (`/login`) with card UI
- [x] Email + password signup form with validation
- [x] "Sign in with Google" button → OAuth2 PKCE flow
- [x] "Sign in with GitHub" button → OAuth2 PKCE flow
- [x] "Sign in with Hugging Face" button → OAuth2 flow
- [x] "Forgot password" link → password reset flow
- [ ] Terms of Service + Privacy Policy agreement checkbox
- [x] Auto-redirect to dashboard after successful login
- [x] JWT token stored in `sessionStorage` (not localStorage)
- [x] **Backend needed**: `POST /api/auth/login` ✅, `POST /api/auth/register` ✅, `POST /api/auth/oauth/{provider}` ✅, `POST /api/auth/refresh` ✅, `POST /api/auth/password-reset` ✅

### UI-1.2 — Session Management
- [x] "Logged in as [name/email]" indicator in navbar
- [x] User avatar (Gravatar or OAuth profile picture)
- [x] Logout button in user dropdown
- [x] Session timeout with "Session expired" modal
- [x] Auto-refresh JWT token before expiry
- [x] All API calls include `Authorization: Bearer <token>` header
- [x] Unauthenticated state → redirect to `/login`

### UI-1.3 — Onboarding Flow (Redesign)
- [x] Replace localStorage-only onboarding modal with server-persisted role selection
- [x] First-login wizard: "I want to..." → Provider / Submitter / Both
- [x] Show relevant tabs based on role (hide admin-only features)
- [ ] "Complete your profile" checklist after signup

---

## Phase UI-2: User Profile & Account Settings (CRITICAL)

### UI-2.1 — Profile Page (`/settings/profile`)
- [x] Display name (editable)
- [x] Email (from OAuth or registration)
- [x] Avatar (from OAuth provider or initial letter)
- [x] Role indicator (Provider / Submitter / Admin)
- [x] Account created date (member since)
- [x] Customer ID / Provider ID display
- [x] Province/country selection (jurisdiction)
- [x] "Delete Account" danger button

### UI-2.2 — API Keys Management (`/settings/api-keys`)
- [x] "Generate API Key" button → calls `POST /api/keys/generate`
- [x] API key list (name, created date, last used, partial key preview)
- [x] Copy-to-clipboard for new keys (shown once)
- [x] Revoke/delete key button per row → `DELETE /api/keys/{preview}`
- [ ] Key permission scopes (read-only, full-access)
- [ ] Rate limit indicator per key

### UI-2.3 — SSH Keys Management (`/settings/ssh-keys`)
- [x] "Generate SSH Key" button → calls `POST /ssh/keygen`
- [x] Upload existing public key form
- [x] SSH key list (name, fingerprint, created date)
- [x] Delete key button
- [x] Show public key → calls `GET /ssh/pubkey`

### UI-2.4 — Notification Settings (`/settings/notifications`)
- [x] Email notifications toggle (job complete, credits low, SLA violation)
- [x] Telegram integration → calls `PUT /alerts/config`
- [x] Browser push notifications toggle
- [ ] Mining alert severity level
- [x] Digest frequency (instant / daily / weekly)

### UI-2.5 — Privacy & Consent (`/settings/privacy`)
- [x] Data collection consent toggles → calls `POST /api/privacy/consent`
  - Cross-border data transfer
  - Telemetry collection
  - Usage profiling
- [x] Current consents display → calls `GET /api/privacy/consent/{entity_id}`
- [x] Revoke consent buttons → calls `DELETE /api/privacy/consent/{entity_id}/{type}`
- [x] Data retention policy display → calls `GET /api/privacy/retention-policies`
- [x] "Request Data Export" button (PIPEDA right) → calls `GET /api/auth/me/data-export`
- [x] "Delete My Data" button → calls `DELETE /api/auth/me`

---

## Phase UI-3: Credits, Wallet & Billing (CRITICAL)

### UI-3.1 — Wallet & Credits Display (Dashboard Header)
- [x] Persistent credit balance in navbar: "$XX.XX Credits" with icon
- [x] Color coding: green (>$50), yellow ($10-$50), red (<$10)
- [x] "+ Add Credits" button always visible
- [x] Low balance warning banner when < $5

### UI-3.2 — Add Credits / Deposit Modal
- [x] Predefined amounts: $10, $25, $50, $100, $250, Custom
- [ ] Payment method selection (Stripe checkout)
- [ ] Auto-reload toggle ("Reload $25 when balance drops below $5")
- [ ] Promo code / coupon field
- [x] Calls `POST /api/billing/wallet/{customer_id}/deposit`

### UI-3.3 — Transaction History (`/billing/history`)
- [x] Full transaction table → calls `GET /api/billing/wallet/{customer_id}/history`
  - Date, Description, Amount (+/-), Balance after
- [x] Filter by type: deposits, charges, refunds, credits
- [x] Date range picker
- [x] CSV export button
- [x] Running balance chart (ASCII art trend)

### UI-3.4 — Invoices (`/billing/invoices`)
- [x] Invoice list with download buttons → calls `GET /api/billing/invoice/{customer_id}`
- [ ] PDF/CSV export per invoice
- [x] CAF (AI Compute Access Fund) rebate column
- [x] GST/HST breakdown per invoice
- [ ] "Export Rebate Docs" button (already exists, keep)

### UI-3.5 — Usage Summary (`/billing/usage`)
- [x] Per-job cost breakdown → calls `GET /api/billing/usage/{customer_id}`
- [ ] Usage by GPU model pie chart
- [ ] Usage by province bar chart
- [ ] Daily/weekly/monthly spend trend line
- [ ] Cost comparison: on-demand vs reserved savings

### UI-3.6 — Refund Status
- [ ] Pending refund display (auto-refunds for host failures)
- [ ] Refund history table → references `POST /api/billing/refund` results
- [ ] Refund reason display (host_failure vs user_error distinction)

---

## Phase UI-4: Reputation Tiers & Trust Badges (HIGH)

### UI-4.1 — 6-Tier Badge System
- [x] Design badge icons/colors for all 6 tiers:
  - 🔘 **New User** — gray, no border, "New" label
  - 🥉 **Bronze** — bronze/brown, subtle ring
  - 🥈 **Silver** — silver gradient, metallic ring
  - 🥇 **Gold** — gold gradient, glowing ring
  - 💎 **Platinum** — platinum/ice blue, diamond sparkle
  - 👑 **Diamond** — purple/rainbow gradient, crown icon, particle effect
- [x] Add CSS classes `.rep-new-user` and `.rep-diamond` (missing from current 4-tier CSS)
- [x] Badge appears: host tables, marketplace listings, leaderboard, user profile
- [ ] Tooltip on hover: "Gold Tier — Score: 523 — 85 jobs completed"

### UI-4.2 — My Reputation Page (`/reputation`)
- [x] Current tier badge (large, centered)
- [x] Score number with progress bar to next tier
- [x] "Next tier in X points" indicator
- [x] Score breakdown: jobs completed, uptime bonus, penalties, decay → calls `GET /api/reputation/{entity_id}/breakdown`
- [x] Score history table → calls `GET /api/reputation/{entity_id}/history`
- [x] Verification badges earned:
  - ✅ Email Verified
  - ✅ Phone Verified
  - ✅ Hardware Audited
  - ✅ Data Center Verified
  - ✅ Incorporated
  - ✅ Government ID
- [ ] "Get Verified" buttons for unearned badges → calls `POST /api/reputation/verify`

### UI-4.3 — Tier Benefits Display
- [x] Table showing benefits per tier:
  - Search boost multiplier
  - Pricing premium allowed
  - Platform commission rate (15% → 8% for Diamond)
  - SLA tier eligibility
- [ ] "How to rank up" guide section

### UI-4.4 — Leaderboard Enhancement
- [ ] Current leaderboard exists — add badge icons next to each entry
- [ ] Rank change arrows (↑3, ↓1, — stable)
- [ ] Click-through to public provider profile
- [ ] Filter by region, GPU type

---

## Phase UI-5: Pricing & Reserved Plans (HIGH)

### UI-5.1 — Pricing Page (`/pricing`)
- [ ] GPU pricing table (exists as "GPU Reference Pricing" — expand to full page)
- [ ] On-demand vs Spot vs Reserved column comparison
- [ ] Availability indicator per GPU model
- [ ] "Rent Now" button per GPU → pre-fills Submit Job form
- [ ] Region/province filter for Canada-only pricing

### UI-5.2 — Reserved Plans
- [x] Reserved plan cards → calls `GET /api/pricing/reserved-plans`:
  - 1-month commitment: 20% off
  - 3-month commitment: 30% off
  - 1-year commitment: 45% off
- [x] "Subscribe" button per plan → calls `POST /api/pricing/reserve`
- [ ] Active commitments display
- [ ] Savings calculator: "You've saved $X with reserved pricing"

### UI-5.3 — Spot Pricing Enhancement
- [ ] Spot price history chart (last 24h, 7d, 30d)
- [x] "Place Spot Bid" form → calls `POST /spot/job`
  - GPU model, max bid $/hr, duration, Docker image
- [x] Current spot prices table (exists — add bid button)
- [ ] Preemption risk indicator (high/medium/low)

---

## Phase UI-6: Job Management Enhancement (HIGH)

### UI-6.1 — Job Detail View (Click-Through)
- [x] Click job row → slide-out panel or modal
- [x] Job metadata: ID, name, status, GPU, priority tier, trust tier, timestamps
- [x] Cost breakdown: rate, duration, total, refund status
- [x] Log tail (last 50 lines) → calls `GET /jobs/{job_id}/logs`
- [x] "Stream Logs" button → opens Job Logs tab pre-filled
- [x] Residency trace → calls `GET /api/jurisdiction/residency-trace/{job_id}`
- [x] Artifacts list → calls `GET /api/artifacts/{job_id}` with download links
- [x] Event audit trail → calls `GET /api/audit/job/{job_id}`
- [x] "Cancel Job" button → calls `PATCH /job/{job_id}` with status=cancelled
- [x] "Requeue Job" button → calls `POST /job/{job_id}/requeue`

### UI-6.2 — Submit Job Form Enhancement
- [x] Multi-GPU selector: `--gpus N` (1, 2, 4, 8)
- [x] Spot vs On-demand toggle with price comparison
- [ ] Docker image browser/autocomplete (popular images)
- [x] NFS mount configuration (server, path, mount point)
- [ ] Job duration estimate
- [x] "Advanced Options" collapsible section

### UI-6.3 — Job Status Improvements
- [x] Status badge colors: queued (blue), running (green), completed (gray), failed (red), cancelled (yellow)
- [x] Progress indicator for running jobs (elapsed time / estimated total)
- [ ] Auto-refresh running jobs (SSE-driven)
- [x] Toast notification on job completion/failure
- [x] Job cancel button in the jobs list table (not just detail view)

---

## Phase UI-7: Host/Provider Management (HIGH)

### UI-7.1 — Host Detail View (Click-Through)
- [x] Click host row → detail panel
- [ ] GPU telemetry gauges (utilization, temp, memory, power)
- [x] SLA status → calls `GET /api/sla/{host_id}`
  - 30-day rolling uptime percentage
  - SLA tier (Community/Secure/Sovereign)
  - Violation count
- [x] Reputation tier badge + score
- [x] Verification status (7 checks with ✅/❌)
- [ ] Earnings for this host → filtered from billing
- [x] Compute score → calls `GET /compute-score/{host_id}`
- [x] "Remove Host" button → calls `DELETE /host/{host_id}`
- [x] "List on Marketplace" / "Unlist" toggle → calls `POST /marketplace/list` / `DELETE /marketplace/{host_id}`

### UI-7.2 — Provider Dashboard Enhancement
- [x] Provider registration modal fix: wrong URL (`/api/stripe/` → `/api/providers/register`) **FIXED**
- [x] Provider detail view → calls `GET /api/providers/{provider_id}`
- [x] Provider earnings display → calls `GET /api/providers/{provider_id}/earnings`
- [x] "Request Payout" button → calls `POST /api/providers/{provider_id}/payout`
- [ ] Incorporation document upload → calls `POST /api/providers/{provider_id}/incorporation`
- [x] GST/HST threshold status → calls `GET /api/billing/gst-threshold/{provider_id}`
- [ ] Platform commission rate display (per reputation tier)

### UI-7.3 — Marketplace Enhancement
- [x] Marketplace search with filters (GPU, price range, location, reputation tier)
- [x] Sort by: price, reputation, availability, compute score
- [x] Marketplace stats summary → calls `GET /marketplace/stats`
- [ ] "Rent This Host" quick button → pre-fills Submit Job

---

## Phase UI-8: SLA & Compliance (MEDIUM)

### UI-8.1 — SLA Dashboard Tab
- [x] SLA targets table → calls `GET /api/sla/targets`
  - Community: 99.0%, Secure: 99.5%, Sovereign: 99.9%
- [x] Per-host SLA status cards → calls `GET /api/sla/{host_id}`
- [x] Violation history table → calls `GET /api/sla/violations/{host_id}`
- [x] Active downtimes list → calls `GET /api/sla/downtimes`
- [x] Credit calculation display (10%/25%/100% based on uptime bracket)
- [x] "Enforce SLA" admin button → calls `POST /api/sla/enforce`

### UI-8.2 — Compliance Dashboard
- [x] Province compliance matrix → calls `GET /api/compliance/provinces`
- [x] Tax rates table → calls `GET /api/compliance/tax-rates`
- [x] Trust tier requirements matrix → calls `GET /api/compliance/trust-tier-requirements`
- [x] Quebec Law 25 PIA check form → calls `POST /api/compliance/quebec-pia-check`
- [x] GST/HST threshold overview → calls `GET /api/billing/gst-threshold`

---

## Phase UI-9: Artifacts & Storage (MEDIUM)

### UI-9.1 — Artifacts Tab
- [x] Artifacts list per job → calls `GET /api/artifacts/{job_id}`
- [x] Upload artifact form → calls `POST /api/artifacts/upload` (presigned URL)
- [x] Download artifact button → calls `POST /api/artifacts/download` (presigned URL)
- [x] Artifact types: job_output, model_checkpoint, dataset, log_bundle
- [x] Residency policy indicator (canada_only, global, region_locked)
- [x] Storage usage display (B2 CA East vs R2)
- [x] Expiry/cleanup date per artifact → calls `GET /api/artifacts/{job_id}/expiry`

---

## Phase UI-10: Analytics & Usage (MEDIUM)

### UI-10.1 — Analytics Dashboard Tab
- [x] Usage analytics → calls `GET /api/analytics/usage`
- [x] Time-series chart: compute hours over time (day/week/month)
- [ ] Breakdown by GPU model (pie chart)
- [x] Breakdown by province (bar chart)
- [x] Canadian vs international compute ratio
- [ ] Cost trend line with forecast
- [ ] Top providers / top consumers tables

---

## Phase UI-11: Admin Panel (MEDIUM)

### UI-11.1 — Admin Tab (Role-Gated)
- [x] Only visible when user role = admin
- [x] System-wide toggles:
  - Canada-only mode → calls `GET/PUT /canada`
  - Maintenance mode
- [x] Alert configuration → calls `GET/PUT /alerts/config`
- [x] Audit chain verification → calls `GET /api/audit/verify-chain`
- [x] Legal request management:
  - Record new → calls `POST /api/transparency/legal-request`
  - Respond → calls `POST /api/transparency/legal-request/{id}/respond`
- [x] Spot price update trigger → calls `POST /spot-prices/update`
- [x] Preemption cycle trigger → calls `POST /spot/preemption-cycle`
- [x] Manual scale → calls `POST /autoscale/up` / `POST /autoscale/down`
- [x] Docker image builder → calls `POST /build`, `GET /builds`
- [x] Metrics snapshot → calls `GET /metrics`

### UI-11.2 — Role-Based UI Gating
- [x] Submitter sees: Jobs, Billing, Wallet, Artifacts, Settings
- [x] Provider sees: My Hosts, My Earnings, Marketplace, Reputation, Settings
- [x] Admin sees: Everything + Admin tab
- [x] Tab visibility controlled by JWT role claim
- [ ] API calls respect role authorization

---

## Phase UI-12: Real-Time & Notifications (LOW)

### UI-12.1 — Toast Notifications
- [x] Bottom-right toast stack for real-time events
- [x] Job completed → green toast with job name
- [x] Job failed → red toast with error summary
- [x] Mining alert → red alert banner (not just event log)
- [x] Credits low → yellow warning toast (moderate) + red critical alert
- [x] SLA violation → orange toast
- [x] Sound toggle for critical alerts (Web Audio API)

### UI-12.2 — Event Log Enhancement
- [x] Filter by event type (job, billing, security, telemetry)
- [x] Search within events
- [x] Severity color coding (info/warning/error/critical)
- [x] Click event → expand details

---

## Phase UI-13: Slurm / HPC Integration (LOW)

### UI-13.1 — Slurm Tab (Admin/Power-User)
- [x] Cluster profile selector → calls `GET /api/slurm/profiles`
- [x] Submit Slurm job form → calls `POST /api/slurm/submit`
- [x] Slurm job status display → calls `GET /api/slurm/status/{id}`
- [x] Cancel Slurm job → calls `DELETE /api/slurm/{id}`
- [x] Nibi/Graham/Narval cluster info

---

## Phase UI-14: Visual Polish & Responsive Design (LOW)

### UI-14.1 — Design System
- [ ] Consistent color palette (dark theme primary)
- [ ] Typography scale
- [ ] Component library (buttons, cards, tables, modals, forms, badges)
- [x] Loading skeleton states (CSS skeleton-pulse animation)
- [x] Empty states with icons and descriptions
- [ ] 404 / error pages

### UI-14.2 — Responsive Layout
- [x] Mobile-friendly navigation (hamburger menu)
- [x] Responsive tables (card view on mobile)
- [x] Touch-friendly modals
- [ ] Viewport-aware charts

---

## Implementation Priority Order

| Priority | Phase | Effort | Impact |
|----------|-------|--------|--------|
| 🔴 P0 | UI-1: Auth + Login (Google/GitHub/HuggingFace/Email) | Large | Blocks everything |
| 🔴 P0 | UI-3.1-3.2: Wallet + Add Credits | Medium | Users can't pay |
| 🔴 P0 | UI-7.2: Fix provider registration bug | Small | Provider reg is broken |
| 🟡 P1 | UI-2.1-2.2: Profile + API Keys | Medium | Account management |
| 🟡 P1 | UI-4.1: 6-Tier badge CSS + icons | Small | Visual identity |
| 🟡 P1 | UI-6.1: Job detail view + cancel | Medium | Core UX |
| 🟡 P1 | UI-3.3-3.4: Transaction history + invoices | Medium | Billing UX |
| 🟡 P1 | UI-7.1: Host detail + remove button | Medium | Provider UX |
| 🟢 P2 | UI-5: Pricing page + reserved plans | Medium | Revenue |
| 🟢 P2 | UI-4.2-4.3: Reputation page + verification badges | Medium | Trust system |
| 🟢 P2 | UI-8: SLA dashboard | Medium | Compliance |
| 🟢 P2 | UI-10: Analytics charts | Medium | Insights |
| 🟢 P2 | UI-6.2-6.3: Submit Job enhancements | Medium | UX polish |
| 🔵 P3 | UI-9: Artifacts/storage | Medium | Completeness |
| 🔵 P3 | UI-2.3-2.5: SSH keys, notifications, privacy | Medium | Settings |
| 🔵 P3 | UI-11: Admin panel + role gating | Large | Governance |
| 🔵 P3 | UI-12: Toast notifications | Small | UX polish |
| ⚪ P4 | UI-13: Slurm integration | Medium | HPC niche |
| ⚪ P4 | UI-14: Responsive + design system | Large | Polish |

---

## Quick Wins (Can Implement Now Without New Backend)

These items use existing API endpoints and only need frontend work:

1. **Fix provider registration URL bug** — change `/api/stripe/provider/register` → `/api/providers/register`
2. **Add `.rep-new-user` and `.rep-diamond` CSS classes** — 2 missing-tier badge styles
3. **Add "Remove Host" button** to host table rows → `DELETE /host/{host_id}`
4. **Add "Cancel Job" button** to job table rows → `PATCH /job/{job_id}`
5. **Add wallet balance display** in header → `GET /api/billing/wallet/{customer_id}`
6. **Add deposit modal** → `POST /api/billing/wallet/{customer_id}/deposit`
7. **Add job detail click-through** → `GET /job/{job_id}`
8. **Add SLA targets table** to Trust tab → `GET /api/sla/targets`
9. **Add analytics summary** → `GET /api/analytics/usage`
10. **Add marketplace list/unlist buttons** → `POST /marketplace/list` / `DELETE /marketplace/{host_id}`

---

## Backend Endpoints Needed (New)

These UI features require new API endpoints:

| Endpoint | Purpose | Phase |
|----------|---------|-------|
| `POST /api/auth/login` | Email/password login | UI-1 |
| `POST /api/auth/register` | Email/password signup | UI-1 |
| `POST /api/auth/oauth/google` | Google OAuth callback | UI-1 |
| `POST /api/auth/oauth/github` | GitHub OAuth callback | UI-1 |
| `POST /api/auth/oauth/huggingface` | HuggingFace OAuth callback | UI-1 |
| `POST /api/auth/refresh` | Refresh JWT token | UI-1 |
| `POST /api/auth/password-reset` | Initiate password reset | UI-1 |
| `GET /api/auth/me` | Get current user profile | UI-2 |
| `PATCH /api/auth/me` | Update profile | UI-2 |
| `DELETE /api/auth/me` | Delete account | UI-2 |
| `GET /api/keys` | List API keys | UI-2 |
| `DELETE /api/keys/{key_id}` | Revoke API key | UI-2 |

---

*Total TODO items: ~180 across 14 phases*
*Estimated effort: 6-8 weeks full implementation*
