# Untested endpoints — coverage-gap worklist

_Regenerated 2026-09-21 by `scripts/regenerate_untested_endpoints.py`: a route/CLI command counts as covered when **either** its path (prefix before `{…}`) **or** its handler function name appears anywhere under `tests/`._ CLI commands require a `test_cmd_*` smoke test in `tests/test_cli*.py`._

**0 of 529 routes (0%)** and **0 of 48 CLI commands** (0%) have no test signal.

Workflow per item: write a `TestClient` (or CLI) test → if it works, tick the box; if it 500s/throws, fix-or-delete then tick. Caveat: a few may be exercised transitively; confirm with the test.

> **What the count above does and does not mean.** A route is scored covered when its path prefix *or* its handler function name appears anywhere under `tests/`. That is a worklist heuristic: it measures **mention**, not **execution**, and the two diverge. Measured against a coverage run of the whole suite, **450 of 525 handlers are actually entered — 78 are never reached**, every one of them scored covered here.

> A test can name a route and only ever receive a 422, because the request body was rejected before the handler ran: that proves the route is mounted and its model validates, and nothing about the code inside. Five tests in `tests/test_untested_endpoints_coverage.py` were exactly that until it was measured. The same blindness let `GET /marketplace/search` be reported tested for as long as this file has existed, because the v2 POST handler shared its function name.

> Reproduce with `scripts/measure_route_execution.py`, which lists the 78. Largest clusters: the whole MFA surface (9), serverless (~20), host admission (5).

## Routes (0 untested)

### `routes/action_plans.py` (0 untested)
- [x] `POST /api/v1/launch-plans` — `api_create_launch_plan`
- [x] `GET /api/v1/launch-plans/{plan_id}` — `api_get_launch_plan`
- [x] `POST /api/v1/launch-plans/{plan_id}/approve` — `api_approve_launch_plan`
- [x] `POST /api/v1/launch-plans/{plan_id}/execute` — `api_execute_launch_plan`
- [x] `POST /api/v1/launch-plans/{plan_id}/revoke` — `api_revoke_launch_plan`
- [x] `POST /api/v1/pipelines` — `api_create_pipeline`
- [x] `GET /api/v1/pipelines/{plan_id}` — `api_get_pipeline`
- [x] `POST /api/v1/pipelines/{plan_id}/execute` — `api_execute_pipeline`
- [x] `POST /api/v1/placements/evaluate` — `api_evaluate_placement`
- [x] `POST /api/v1/placements/simulate` — `api_simulate_placement`

### `routes/admin.py` (0 untested)
- [x] `GET /api/admin/activity` — `api_admin_activity`  ✓ test_admin_endpoints_coverage.py
- [x] `GET /api/admin/agent-tokens/coverage` — `api_admin_agent_token_coverage`
- [x] `POST /api/admin/agent/rollout` — `api_admin_agent_rollout`  ✓ test_admin_endpoints_coverage.py
- [x] `GET /api/admin/ai-conversations` — `api_admin_ai_conversations`  ✓ test_admin_endpoints_coverage.py
- [x] `GET /api/admin/ai-stats` — `api_admin_ai_stats`  ✓ test_admin_endpoints_coverage.py
- [x] `GET /api/admin/control-plane/jobs` — `api_admin_control_plane_jobs`  ✓ test_admin_endpoints_coverage.py
- [x] `GET /api/admin/control-plane/scheduled-tasks` — `api_admin_control_plane_scheduled_tasks`  ✓ test_admin_endpoints_coverage.py
- [x] `GET /api/admin/hosts/{host_id}/agent-tokens` — `api_admin_list_host_agent_tokens`
- [x] `POST /api/admin/hosts/{host_id}/agent-tokens` — `api_admin_issue_host_agent_token`
- [x] `POST /api/admin/hosts/{host_id}/agent-tokens/revoke` — `api_admin_revoke_host_agent_tokens`
- [x] `GET /api/admin/infrastructure` — `api_admin_infrastructure`  ✓ test_admin_endpoints_coverage.py
- [x] `GET /api/admin/overview` — `api_admin_overview`  ✓ test_admin_endpoints_coverage.py
- [x] `GET /api/admin/reconciler/findings` — `api_admin_reconciler_findings`  ✓ test_admin_endpoints_coverage.py
- [x] `POST /api/admin/reconciler/findings/{finding_id}/dismiss` — `api_admin_reconciler_dismiss`
- [x] `POST /api/admin/reconciler/findings/{finding_id}/enforce` — `api_admin_reconciler_enforce`
- [x] `POST /api/admin/reconciler/reconcile-host/{host_id}` — `api_admin_reconcile_host`  ✓ test_admin_endpoints_coverage.py
- [x] `GET /api/admin/revenue` — `api_admin_revenue`  ✓ test_admin_endpoints_coverage.py
- [x] `GET /api/admin/stats` — `api_admin_stats`
- [x] `GET /api/admin/teams` — `api_admin_teams`  ✓ test_admin_endpoints_coverage.py
- [x] `DELETE /api/admin/teams/{team_id}/members/{email}` — `api_admin_remove_team_member`  ✓ test_admin_endpoints_coverage.py
- [x] `GET /api/admin/unit-economics` — `api_admin_unit_economics`  ✓ test_admin_endpoints_coverage.py
- [x] `GET /api/admin/users` — `api_admin_users`  ✓ test_admin_endpoints_coverage.py
- [x] `POST /api/admin/users/{email}/role` — `api_admin_set_user_role`  ✓ test_admin_endpoints_coverage.py
- [x] `POST /api/admin/users/{email}/toggle-admin` — `api_admin_toggle_admin`  ✓ test_admin_endpoints_coverage.py
- [x] `GET /api/admin/verification-queue` — `api_admin_verification_queue`  ✓ test_admin_endpoints_coverage.py
- [x] `POST /api/admin/web-push/test-notification` — `api_admin_web_push_test_notification`

### `routes/agent.py` (0 untested)
- [x] `POST /agent/benchmark` — `api_agent_benchmark`
- [x] `GET /agent/commands/{host_id}` — `api_agent_commands_drain`
- [x] `POST /agent/degraded` — `api_agent_degraded`  ✓ test_untested_endpoints_coverage.py
- [x] `POST /agent/lease/claim` — `api_agent_lease_claim`
- [x] `POST /agent/lease/release` — `api_agent_lease_release`
- [x] `POST /agent/lease/renew` — `api_agent_lease_renew`
- [x] `POST /agent/logs/{job_id}` — `api_agent_logs`
- [x] `POST /agent/mining-alert` — `api_mining_alert`
- [x] `GET /agent/popular-images` — `api_agent_popular_images`
- [x] `GET /agent/preempt/{host_id}` — `api_agent_preempt`
- [x] `POST /agent/preempt/{host_id}/{job_id}` — `api_schedule_preemption`
- [x] `GET /agent/ssh-keys/{job_id}` — `api_agent_ssh_keys`
- [x] `POST /agent/ssh-status/{job_id}` — `api_agent_ssh_status`  ✓ test_agent_endpoints_coverage.py
- [x] `POST /agent/telemetry` — `api_agent_telemetry`
- [x] `GET /agent/telemetry/{host_id}` — `api_get_telemetry`
- [x] `POST /agent/versions` — `api_agent_versions`
- [x] `GET /agent/work/{host_id}` — `api_agent_work`
- [x] `GET /api/telemetry/all` — `api_all_telemetry`

### `routes/agent_v2.py` (0 untested)
- [x] `POST /agent/v2/attempts/status` — `api_v2_attempt_status`
- [x] `POST /agent/v2/commands/claim` — `api_v2_commands_claim`
- [x] `POST /agent/v2/commands/{command_id}/ack` — `api_v2_command_ack`
- [x] `POST /agent/v2/commands/{command_id}/nack` — `api_v2_command_nack`
- [x] `POST /agent/v2/leases/claim` — `api_v2_lease_claim`
- [x] `POST /agent/v2/leases/release` — `api_v2_lease_release`
- [x] `POST /agent/v2/leases/renew` — `api_v2_lease_renew`
- [x] `GET /agent/v2/negotiate/{host_id}` — `api_v2_negotiate`
- [x] `POST /agent/v2/observations` — `api_v2_observations`
- [x] `POST /agent/v2/tokens/rotate` — `api_v2_token_rotate`

### `routes/artifacts.py` (0 untested)
- [x] `GET /api/artifacts` — `api_list_all_artifacts`  ✓ test_untested_endpoints_coverage.py
- [x] `POST /api/artifacts/download` — `api_request_download`  ✓ test_artifacts_endpoints_coverage.py
- [x] `POST /api/artifacts/finalize` — `api_finalize_upload`  ✓ test_untested_endpoints_coverage.py
- [x] `POST /api/artifacts/upload` — `api_request_upload`
- [x] `GET /api/artifacts/{job_id}` — `api_list_artifacts`  ✓ test_untested_endpoints_coverage.py
- [x] `GET /api/artifacts/{job_id}/expiry` — `api_artifact_expiry`  ✓ test_untested_endpoints_coverage.py

### `routes/auth.py` (0 untested)
- [x] `GET /.well-known/jwks.json` — `oauth_jwks_document`  ✓ test_untested_endpoints_coverage.py
- [x] `GET /.well-known/oauth-authorization-server` — `oauth_authorization_server_metadata`
- [x] `GET /api/agent-keys` — `api_list_agent_keys`  ✓ test_untested_endpoints_coverage.py
- [x] `DELETE /api/agent-keys/{key_id}` — `api_revoke_agent_key`  ✓ test_untested_endpoints_coverage.py
- [x] `PATCH /api/agent-keys/{key_id}` — `api_rename_agent_key`  ✓ test_untested_endpoints_coverage.py
- [x] `POST /api/auth/change-password` — `api_auth_change_password`
- [x] `GET /api/auth/demo-credentials` — `api_auth_demo_credentials`  ✓ test_untested_endpoints_coverage.py
- [x] `POST /api/auth/device` — `oauth_device_authorize_compat`
- [x] `GET /api/auth/introspect` — `api_auth_introspect`
- [x] `POST /api/auth/login` — `api_auth_login`  ✓ test_sla_endpoints_coverage.py
- [x] `POST /api/auth/logout` — `api_auth_logout`  ✓ test_auth_endpoints_coverage.py
- [x] `GET /api/auth/me` — `api_auth_me`  ✓ test_chat_endpoints_coverage.py
- [x] `PATCH /api/auth/me` — `api_auth_update_profile`  ✓ test_chat_endpoints_coverage.py
- [x] `DELETE /api/auth/me` — `api_auth_delete_account`  ✓ test_chat_endpoints_coverage.py
- [x] `POST /api/auth/me/avatar` — `api_auth_upload_avatar`  ✓ test_untested_endpoints_coverage.py
- [x] `GET /api/auth/me/avatar` — `api_auth_get_avatar`  ✓ test_untested_endpoints_coverage.py
- [x] `DELETE /api/auth/me/avatar` — `api_auth_delete_avatar`  ✓ test_untested_endpoints_coverage.py
- [x] `GET /api/auth/me/data-export` — `api_data_export`
- [x] `POST /api/auth/me/email-change` — `api_auth_email_change`  ✓ test_untested_endpoints_coverage.py
- [x] `POST /api/auth/me/email-change/confirm` — `api_auth_email_change_confirm`  ✓ test_untested_endpoints_coverage.py
- [x] `PUT /api/auth/me/profile` — `api_auth_put_profile`  ✓ test_untested_endpoints_coverage.py
- [x] `POST /api/auth/oauth/facebook/deauthorize` — `facebook_deauthorize`
- [x] `POST /api/auth/oauth/facebook/delete-data` — `facebook_delete_data`
- [x] `POST /api/auth/oauth/{provider}` — `api_auth_oauth_initiate`  ✓ test_auth_endpoints_coverage.py
- [x] `GET /api/auth/oauth/{provider}/callback` — `api_auth_oauth_callback`  ✓ test_auth_endpoints_coverage.py
- [x] `POST /api/auth/password-reset` — `api_auth_password_reset`
- [x] `POST /api/auth/password-reset/confirm` — `api_auth_password_reset_confirm`
- [x] `POST /api/auth/refresh` — `api_auth_refresh`
- [x] `POST /api/auth/register` — `api_auth_register`  ✓ test_sla_endpoints_coverage.py
- [x] `POST /api/auth/resend-verification` — `api_auth_resend_verification`  ✓ test_auth_endpoints_coverage.py
- [x] `GET /api/auth/sessions` — `api_auth_list_sessions`  ✓ test_auth_endpoints_coverage.py
- [x] `DELETE /api/auth/sessions/{token_prefix}` — `api_auth_revoke_session`  ✓ test_auth_endpoints_coverage.py
- [x] `POST /api/auth/token` — `oauth_token_compat`
- [x] `GET /api/auth/verify` — `oauth_verify_page`
- [x] `POST /api/auth/verify` — `oauth_verify_device`
- [x] `POST /api/auth/verify-email` — `api_auth_verify_email`
- [x] `GET /api/keys` — `api_list_keys`
- [x] `POST /api/keys/generate` — `api_generate_api_key`
- [x] `DELETE /api/keys/{key_preview}` — `api_revoke_key`
- [x] `GET /api/mcp/quick-connect` — `api_mcp_quick_connect`
- [x] `POST /api/oauth/clients` — `api_create_oauth_client`  ✓ test_auth_endpoints_coverage.py
- [x] `GET /api/oauth/clients` — `api_list_oauth_clients`  ✓ test_auth_endpoints_coverage.py
- [x] `DELETE /api/oauth/clients/{client_id}` — `api_delete_oauth_client`  ✓ test_auth_endpoints_coverage.py
- [x] `PATCH /api/oauth/clients/{client_id}` — `api_update_oauth_client`  ✓ test_auth_endpoints_coverage.py
- [x] `POST /api/oauth/clients/{client_id}/rotate-secret` — `api_rotate_oauth_client_secret`  ✓ test_auth_endpoints_coverage.py
- [x] `GET /api/users/me/preferences` — `api_get_user_preferences`  ✓ test_auth_endpoints_coverage.py
- [x] `PUT /api/users/me/preferences` — `api_set_user_preferences`  ✓ test_auth_endpoints_coverage.py
- [x] `GET /oauth/authorize` — `oauth_authorize`
- [x] `POST /oauth/authorize` — `oauth_authorize_decision`
- [x] `POST /oauth/device/authorize` — `oauth_device_authorize`  ✓ test_auth_endpoints_coverage.py
- [x] `POST /oauth/register` — `oauth_dynamic_client_registration`
- [x] `POST /oauth/token` — `oauth_token`  ✓ test_hosts_endpoints_coverage.py

### `routes/autoscale.py` (0 untested)
- [x] `POST /autoscale/cycle` — `api_autoscale_cycle`  ✓ test_autoscale_endpoints_coverage.py
- [x] `POST /autoscale/down` — `api_autoscale_down`  ✓ test_autoscale_endpoints_coverage.py
- [x] `POST /autoscale/pool` — `api_add_to_pool`  ✓ test_autoscale_endpoints_coverage.py
- [x] `GET /autoscale/pool` — `api_get_pool`  ✓ test_autoscale_endpoints_coverage.py
- [x] `DELETE /autoscale/pool/{host_id}` — `api_remove_from_pool`
- [x] `POST /autoscale/up` — `api_autoscale_up`  ✓ test_autoscale_endpoints_coverage.py

### `routes/billing.py` (0 untested)
- [x] `GET /api/analytics/enhanced` — `api_analytics_enhanced`
- [x] `GET /api/analytics/usage` — `api_usage_analytics`
- [x] `GET /api/billing/attestation` — `api_provider_attestation`
- [x] `POST /api/billing/crypto/deposit` — `api_crypto_deposit`  ✓ test_billing_endpoints_coverage.py
- [x] `GET /api/billing/crypto/deposit/{deposit_id}` — `api_crypto_deposit_status`  ✓ test_billing_endpoints_coverage.py
- [x] `GET /api/billing/crypto/enabled` — `api_crypto_enabled`
- [x] `GET /api/billing/crypto/rate` — `api_crypto_rate`  ✓ test_billing_endpoints_coverage.py
- [x] `POST /api/billing/crypto/refresh/{deposit_id}` — `api_crypto_refresh`  ✓ test_billing_endpoints_coverage.py
- [x] `POST /api/billing/free-credits/{customer_id}` — `api_claim_free_credits`
- [x] `GET /api/billing/free-credits/{customer_id}/status` — `api_free_credits_status`
- [x] `GET /api/billing/invoice/{customer_id}` — `api_generate_invoice`  ✓ test_billing_endpoints_coverage.py
- [x] `GET /api/billing/invoice/{customer_id}/download` — `api_download_invoice`  ✓ test_billing_endpoints_coverage.py
- [x] `GET /api/billing/invoices/{customer_id}` — `api_list_invoices`  ✓ test_billing_endpoints_coverage.py
- [x] `POST /api/billing/lightning/deposit` — `api_ln_create_deposit`  ✓ test_billing_endpoints_coverage.py
- [x] `GET /api/billing/lightning/deposit/{deposit_id}` — `api_ln_check_deposit`  ✓ test_billing_endpoints_coverage.py
- [x] `GET /api/billing/lightning/enabled` — `api_ln_enabled`  ✓ test_billing_endpoints_coverage.py
- [x] `GET /api/billing/lightning/rate` — `api_ln_rate`  ✓ test_billing_endpoints_coverage.py
- [x] `POST /api/billing/payment-intent` — `api_create_payment_intent`  ✓ test_billing_endpoints_coverage.py
- [x] `GET /api/billing/payment-methods` — `api_billing_list_payment_methods`  ✓ test_billing_endpoints_coverage.py
- [x] `DELETE /api/billing/payment-methods/{payment_method_id}` — `api_billing_detach_payment_method`  ✓ test_billing_endpoints_coverage.py
- [x] `POST /api/billing/paypal/capture-order` — `api_paypal_capture_order`
- [x] `POST /api/billing/paypal/create-order` — `api_paypal_create_order`
- [x] `GET /api/billing/paypal/enabled` — `api_paypal_enabled`  ✓ test_billing_endpoints_coverage.py
- [x] `POST /api/billing/paypal/marketplace/capture-order` — `api_paypal_marketplace_capture_order`
- [x] `POST /api/billing/paypal/marketplace/create-order` — `api_paypal_marketplace_create_order`
- [x] `POST /api/billing/paypal/webhook` — `api_paypal_webhook`
- [x] `POST /api/billing/portal-session` — `api_billing_portal_session`
- [x] `POST /api/billing/refund` — `api_process_refund`
- [x] `POST /api/billing/setup-intent` — `api_billing_setup_intent`  ✓ test_billing_endpoints_coverage.py
- [x] `GET /api/billing/usage/{customer_id}` — `api_usage_summary`  ✓ test_billing_endpoints_coverage.py
- [x] `GET /api/billing/wallet/{customer_id}` — `api_get_wallet`  ✓ test_billing_endpoints_coverage.py
- [x] `GET /api/billing/wallet/{customer_id}/depletion` — `api_wallet_depletion`  ✓ test_billing_endpoints_coverage.py
- [x] `POST /api/billing/wallet/{customer_id}/deposit` — `api_deposit`  ✓ test_billing_endpoints_coverage.py
- [x] `GET /api/billing/wallet/{customer_id}/history` — `api_wallet_history`  ✓ test_billing_endpoints_coverage.py
- [x] `POST /api/billing/wallet/{customer_id}/reset-testing` — `api_reset_wallet_testing_state`  ✓ test_billing_endpoints_coverage.py
- [x] `POST /api/pricing/estimate` — `api_estimate_cost`
- [x] `GET /api/pricing/models` — `api_pricing_models`  ✓ test_billing_endpoints_coverage.py
- [x] `GET /api/pricing/rates` — `api_pricing_rates`  ✓ test_billing_endpoints_coverage.py
- [x] `GET /api/pricing/reference` — `api_reference_pricing`
- [x] `GET /api/pricing/reservations` — `api_list_reservations`  ✓ test_billing_endpoints_coverage.py
- [x] `POST /api/pricing/reserve` — `api_reserve_commitment`
- [x] `GET /api/pricing/reserved-plans` — `api_reserved_plans`
- [x] `GET /api/pricing/spot-quote` — `api_spot_quote`  ✓ test_untested_endpoints_coverage.py
- [x] `POST /api/v2/billing/auto-topup` — `api_billing_configure_topup`  ✓ test_billing_endpoints_coverage.py
- [x] `GET /api/v2/billing/auto-topup` — `api_billing_get_topup`  ✓ test_billing_endpoints_coverage.py
- [x] `POST /api/v2/billing/auto-topup-plans` — `api_billing_auto_topup_plan`
- [x] `POST /api/v2/billing/auto-topup-plans/{plan_id}/execute` — `api_billing_execute_auto_topup_plan`
- [x] `GET /api/v2/billing/pending-verification` — `api_billing_pending_verification`
- [x] `POST /api/v2/billing/top-up` — `api_billing_manual_topup`
- [x] `GET /billing` — `api_billing`  ✓ test_billing_endpoints_coverage.py
- [x] `POST /billing/bill-all` — `api_bill_all`
- [x] `POST /billing/bill/{job_id}` — `api_bill_instance`

### `routes/chat.py` (0 untested)
- [x] `POST /api/ai/analytics` — `api_ai_analytics_chat`  ✓ test_chat_endpoints_coverage.py
- [x] `POST /api/ai/chat` — `api_ai_chat`  ✓ test_chat_endpoints_coverage.py
- [x] `POST /api/ai/confirm` — `api_ai_confirm`  ✓ test_chat_endpoints_coverage.py
- [x] `GET /api/ai/conversations` — `api_ai_list_conversations`  ✓ test_chat_endpoints_coverage.py
- [x] `GET /api/ai/conversations/{conversation_id}` — `api_ai_get_conversation`  ✓ test_chat_endpoints_coverage.py
- [x] `DELETE /api/ai/conversations/{conversation_id}` — `api_ai_delete_conversation`  ✓ test_chat_endpoints_coverage.py
- [x] `GET /api/ai/suggestions` — `api_ai_suggestions`  ✓ test_chat_endpoints_coverage.py
- [x] `POST /api/chat` — `api_chat`  ✓ test_chat_endpoints_coverage.py
- [x] `GET /api/chat/conversations` — `api_chat_conversations`  ✓ test_chat_endpoints_coverage.py
- [x] `POST /api/chat/feedback` — `api_chat_feedback`  ✓ test_chat_endpoints_coverage.py
- [x] `GET /api/chat/history/{conversation_id}` — `api_chat_history`
- [x] `GET /api/chat/suggestions` — `api_chat_suggestions`

### `routes/cloudburst.py` (0 untested)
- [x] `GET /api/v2/burst/status` — `api_burst_status`  ✓ test_cloudburst_endpoints_coverage.py

### `routes/compliance.py` (0 untested)
- [x] `GET /api/billing/gst-threshold` — `api_gst_threshold_status`  ✓ test_compliance_endpoints_coverage.py
- [x] `GET /api/billing/gst-threshold/{provider_id}` — `api_provider_gst_threshold`  ✓ test_compliance_endpoints_coverage.py
- [x] `GET /api/compliance/status` — `api_compliance_status`
- [x] `GET /api/compliance/tax-rates` — `api_tax_rates`  ✓ test_compliance_endpoints_coverage.py

### `routes/control_plane_v1.py` (0 untested)
- [x] `POST /api/v1/control-plane/commands/{command_id}/retry` — `api_v1_retry_agent_command`
- [x] `GET /api/v1/control-plane/health` — `api_v1_control_plane_health`
- [x] `GET /api/v1/control-plane/queue` — `api_v1_control_plane_queue`
- [x] `GET /api/v1/control-plane/reconciliation-findings` — `api_v1_reconciliation_findings`
- [x] `GET /api/v1/hosts/{host_id}/capacity` — `api_v1_host_capacity`
- [x] `POST /api/v1/hosts/{host_id}/drain` — `api_v1_drain_host`
- [x] `POST /api/v1/hosts/{host_id}/eviction-plans` — `api_v1_create_eviction_plan`
- [x] `POST /api/v1/hosts/{host_id}/eviction-plans/{plan_id}/execute` — `api_v1_execute_eviction_plan`
- [x] `POST /api/v1/hosts/{host_id}/evictions` — `api_v1_evict_host_workloads`
- [x] `GET /api/v1/hosts/{host_id}/observations` — `api_v1_host_observations`
- [x] `POST /api/v1/hosts/{host_id}/undrain` — `api_v1_undrain_host`
- [x] `GET /api/v1/instances/{job_id}` — `api_v1_instance`
- [x] `GET /api/v1/instances/{job_id}/active-lease` — `api_v1_instance_active_lease`
- [x] `GET /api/v1/instances/{job_id}/attempts` — `api_v1_instance_attempts`
- [x] `GET /api/v1/instances/{job_id}/control-plane` — `api_v1_instance_control_plane`
- [x] `GET /api/v1/instances/{job_id}/events` — `api_v1_instance_events`
- [x] `GET /api/v1/instances/{job_id}/placement-explanation` — `api_v1_instance_placement_explanation`
- [x] `POST /api/v1/instances/{job_id}/reconcile` — `api_v1_instance_reconcile`
- [x] `POST /api/v1/instances/{job_id}/retry` — `api_v1_instance_retry`
- [x] `GET /api/v1/instances/{job_id}/timeline` — `api_v1_instance_timeline`
- [x] `GET /api/v1/mcp/activation-funnel` — `api_v1_mcp_activation_funnel`
- [x] `POST /api/v1/mcp/tool-audit` — `api_v1_mcp_tool_audit`
- [x] `GET /api/v1/mcp/tool-audit` — `api_v1_mcp_tool_audit_export`
- [x] `GET /api/v1/openapi.json` — `api_v1_openapi`

### `routes/events.py` (0 untested)
- [x] `GET /api/audit/instance/{job_id}` — `api_instance_audit_trail`  ✓ test_events_endpoints_coverage.py
- [x] `GET /api/audit/verify-chain` — `api_verify_event_chain`  ✓ test_events_endpoints_coverage.py
- [x] `GET /api/events` — `api_get_all_events`  ✓ test_events_endpoints_coverage.py
- [x] `GET /api/events/leases/{job_id}` — `api_get_lease`  ✓ test_events_endpoints_coverage.py
- [x] `GET /api/events/{entity_type}/{entity_id}` — `api_get_events`  ✓ test_events_endpoints_coverage.py

### `routes/gpu.py` (0 untested)
- [x] `GET /api/v2/gpu/available` — `api_gpu_available`

### `routes/health.py` (0 untested)
- [x] `GET /` — `root`  ✓ test_sla_endpoints_coverage.py
- [x] `POST /_internal/legacy-auth/device` — `api_auth_verify_device`  ✓ test_health_endpoints_coverage.py
- [x] `POST /_internal/legacy-auth/token` — `api_auth_verify_page`  ✓ test_health_endpoints_coverage.py
- [x] `POST /_internal/legacy-auth/verify` — `_require_provider_or_admin`  ✓ test_health_endpoints_coverage.py
- [x] `GET /_internal/legacy-auth/verify` — `api_slurm_submit`  ✓ test_health_endpoints_coverage.py
- [x] `GET /alerts/config` — `api_set_alert_config`  ✓ test_health_endpoints_coverage.py
- [x] `PUT /alerts/config` — `api_generate_ssh_key`  ✓ test_health_endpoints_coverage.py
- [x] `GET /api/alerts/config` — `api_get_alert_config_alias`  ✓ test_health_endpoints_coverage.py
- [x] `PUT /api/alerts/config` — `api_set_alert_config_alias`  ✓ test_health_endpoints_coverage.py
- [x] `GET /api/nfs/config` — `api_build_image`
- [x] `GET /api/slurm/instances` — `api_slurm_list_instances`  ✓ test_health_endpoints_coverage.py
- [x] `GET /api/slurm/profiles` — `api_nfs_config`
- [x] `GET /api/slurm/status/{slurm_job_id}` — `api_slurm_cancel`
- [x] `POST /api/slurm/submit` — `api_slurm_status`
- [x] `DELETE /api/slurm/{slurm_job_id}` — `api_slurm_profiles`  ✓ test_health_endpoints_coverage.py
- [x] `GET /api/ssh/pubkey` — `api_auth_device_code`  ✓ test_health_endpoints_coverage.py
- [x] `GET /api/status` — `service_status`
- [x] `GET /api/stream` — `sse_stream`
- [x] `POST /build` — `api_list_builds`  ✓ test_health_endpoints_coverage.py
- [x] `POST /build/{model}/dockerfile` — `_sse_generator`  ✓ test_health_endpoints_coverage.py
- [x] `GET /builds` — `api_generate_dockerfile`  ✓ test_health_endpoints_coverage.py
- [x] `GET /dashboard` — `dashboard`  ✓ test_stripe_connect_v2_endpoints_coverage.py
- [x] `GET /healthz` — `healthz`  ✓ test_health_endpoints_coverage.py
- [x] `GET /legacy` — `legacy_dashboard`  ✓ test_health_endpoints_coverage.py
- [x] `GET /legacy/{path:path}` — `api_get_alert_config`  ✓ test_health_endpoints_coverage.py
- [x] `GET /livez` — `livez`
- [x] `GET /llms.txt` — `api_llms_txt`  ✓ test_health_endpoints_coverage.py
- [x] `GET /metrics` — `metrics`  ✓ test_health_endpoints_coverage.py
- [x] `GET /metrics/prometheus` — `metrics_prometheus`  ✓ test_health_endpoints_coverage.py
- [x] `GET /readyz` — `readyz`
- [x] `POST /ssh/keygen` — `api_get_pubkey`  ✓ test_health_endpoints_coverage.py
- [x] `GET /ssh/pubkey` — `api_generate_token`  ✓ test_health_endpoints_coverage.py
- [x] `GET /startupz` — `startupz`
- [x] `POST /token/generate` — `api_auth_device_token`

### `routes/host_admission.py` (0 untested)
- [x] `GET /api/admin/admission-queue` — `api_admission_queue`  ✓ test_untested_endpoints_coverage.py
- [x] `POST /api/admin/hosts/{host_id}/admission-decisions` — `api_decide_admission`
- [x] `POST /api/admin/hosts/{host_id}/authoritative-evidence` — `api_record_authoritative_evidence`
- [x] `POST /api/hosts/compatibility-sessions/{session_id}/evidence` — `api_submit_compatibility_evidence`  ✓ test_untested_endpoints_coverage.py
- [x] `GET /api/hosts/{host_id}/admission` — `api_admission_status`  ✓ test_untested_endpoints_coverage.py
- [x] `POST /api/hosts/{host_id}/compatibility-sessions` — `api_create_compatibility_session`  ✓ test_untested_endpoints_coverage.py
- [x] `POST /api/hosts/{host_id}/provider-evidence` — `api_record_provider_evidence`  ✓ test_untested_endpoints_coverage.py

### `routes/hosts.py` (0 untested)
- [x] `POST /api/hosts/register` — `api_register_host_web`  ✓ test_hosts_endpoints_coverage.py
- [x] `GET /api/hosts/{host_id}/spot-preview` — `api_host_spot_preview`  ✓ test_untested_endpoints_coverage.py
- [x] `PATCH /api/hosts/{host_id}/spot-settings` — `api_update_host_spot_settings`  ✓ test_untested_endpoints_coverage.py
- [x] `GET /compute-score/{host_id}` — `api_get_compute_score`
- [x] `GET /compute-scores` — `api_list_compute_scores`  ✓ test_hosts_endpoints_coverage.py
- [x] `PUT /host` — `api_register_host`  ✓ test_untested_endpoints_coverage.py
- [x] `GET /host/{host_id}` — `api_get_host`  ✓ test_hosts_endpoints_coverage.py
- [x] `DELETE /host/{host_id}` — `api_remove_host`  ✓ test_hosts_endpoints_coverage.py
- [x] `POST /host/{host_id}/drain` — `api_drain_host`  ✓ test_hosts_endpoints_coverage.py
- [x] `GET /host/{host_id}/maintenance` — `api_host_maintenance`  ✓ test_hosts_endpoints_coverage.py
- [x] `POST /host/{host_id}/undrain` — `api_undrain_host`  ✓ test_hosts_endpoints_coverage.py
- [x] `GET /hosts` — `api_list_hosts`  ✓ test_untested_endpoints_coverage.py
- [x] `POST /hosts/check` — `api_check_hosts`  ✓ test_hosts_endpoints_coverage.py

### `routes/inference.py` (0 untested)
- [x] `POST /api/inference` — `api_inference_submit`
- [x] `GET /api/inference/models/available` — `api_inference_models`
- [x] `GET /api/inference/{job_id}` — `api_inference_result`
- [x] `POST /api/inference/{job_id}/result` — `api_inference_post_result`
- [x] `POST /v1/inference` — `api_v1_inference_sync`
- [x] `POST /v1/inference/async` — `api_v1_inference_async`
- [x] `GET /v1/inference/{job_id}` — `api_v1_inference_poll`

### `routes/instances.py` (0 untested)
- [x] `POST /admin/instances/{job_id}/reinject-shell` — `api_admin_reinject_shell`  ✓ test_instances_endpoints_coverage.py
- [x] `GET /api/images/templates` — `api_image_templates`  ✓ test_instances_endpoints_coverage.py
- [x] `POST /api/instances/{job_id}/stream-ticket` — `api_instance_stream_ticket`
- [x] `GET /api/instances/{job_id}/telemetry` — `api_instance_telemetry`
- [x] `POST /api/v1/image-sweep-plans/{plan_id}/execute` — `api_execute_image_sweep_plan`
- [x] `POST /api/v1/image-sweeps` — `api_create_image_sweep`
- [x] `GET /api/v1/image-sweeps/{sweep_id}` — `api_get_image_sweep`
- [x] `POST /api/v2/scheduler/process-binpack` — `api_process_queue_binpack`
- [x] `POST /failover` — `api_failover`
- [x] `POST /instance` — `api_submit_instance`  ✓ test_health_endpoints_coverage.py
- [x] `GET /instance/{job_id}` — `api_get_instance`  ✓ test_instances_endpoints_coverage.py
- [x] `PATCH /instance/{job_id}` — `api_update_instance`  ✓ test_instances_endpoints_coverage.py
- [x] `PATCH /instance/{job_id}/name` — `api_rename_instance`  ✓ test_instances_endpoints_coverage.py
- [x] `POST /instance/{job_id}/requeue` — `api_requeue_instance`  ✓ test_instances_endpoints_coverage.py
- [x] `GET /instances` — `api_list_instances`  ✓ test_health_endpoints_coverage.py
- [x] `GET /instances/{job_id}/auto-launch` — `api_instances_auto_launch_get`  ✓ test_instances_endpoints_coverage.py
- [x] `POST /instances/{job_id}/auto-launch/report` — `api_instances_auto_launch_report`  ✓ test_instances_endpoints_coverage.py
- [x] `POST /instances/{job_id}/cancel` — `api_cancel_instance`  ✓ test_instances_endpoints_coverage.py
- [x] `POST /instances/{job_id}/expose` — `api_instances_expose`  ✓ test_instances_endpoints_coverage.py
- [x] `POST /instances/{job_id}/http-ports/report` — `api_http_ports_report`  ✓ test_instances_endpoints_coverage.py
- [x] `POST /instances/{job_id}/lock` — `api_lock_instance`  ✓ test_instances_endpoints_coverage.py
- [x] `GET /instances/{job_id}/logs` — `api_instance_logs`  ✓ test_instances_endpoints_coverage.py
- [x] `GET /instances/{job_id}/logs/download` — `api_instance_logs_download`  ✓ test_instances_endpoints_coverage.py
- [x] `GET /instances/{job_id}/logs/stream` — `api_instance_log_stream`  ✓ test_instances_endpoints_coverage.py
- [x] `POST /instances/{job_id}/reset` — `api_reset_instance`  ✓ test_instances_endpoints_coverage.py
- [x] `POST /instances/{job_id}/restart` — `api_restart_instance`  ✓ test_instances_endpoints_coverage.py
- [x] `POST /instances/{job_id}/snapshot` — `api_snapshot_instance`  ✓ test_instances_endpoints_coverage.py
- [x] `POST /instances/{job_id}/start` — `api_start_instance`  ✓ test_instances_endpoints_coverage.py
- [x] `POST /instances/{job_id}/stop` — `api_stop_instance`  ✓ test_instances_endpoints_coverage.py
- [x] `POST /instances/{job_id}/terminate` — `api_terminate_instance`  ✓ test_instances_endpoints_coverage.py
- [x] `POST /instances/{job_id}/unlock` — `api_unlock_instance`  ✓ test_instances_endpoints_coverage.py
- [x] `GET /internal/route/{slug}/{port}` — `api_internal_route`
- [x] `POST /queue/process` — `api_process_queue`
- [x] `GET /tiers` — `api_list_tiers`
- [x] `GET /user-images` — `api_list_user_images`
- [x] `PATCH /user-images/{image_id}` — `api_patch_user_image`
- [x] `DELETE /user-images/{image_id}` — `api_delete_user_image`
- [x] `POST /user-images/{image_id}/complete` — `api_user_image_complete`

### `routes/marketplace.py` (0 untested)
- [x] `POST /api/v2/marketplace/allocate` — `api_marketplace_allocate`
- [x] `POST /api/v2/marketplace/offers` — `api_marketplace_create_offer`
- [x] `POST /api/v2/marketplace/release/{allocation_id}` — `api_marketplace_release`  ✓ test_marketplace_endpoints_coverage.py
- [x] `POST /api/v2/marketplace/reservations` — `api_marketplace_create_reservation`
- [x] `DELETE /api/v2/marketplace/reservations/{reservation_id}` — `api_marketplace_cancel_reservation`
- [x] `POST /api/v2/marketplace/search` — `api_marketplace_search_v2`  ✓ test_marketplace_endpoints_coverage.py
- [x] `GET /api/v2/marketplace/spot-prices` — `api_marketplace_spot_prices`  ✓ test_marketplace_endpoints_coverage.py
- [x] `GET /api/v2/marketplace/spot-prices/{gpu_model}/history` — `api_marketplace_spot_history`  ✓ test_marketplace_endpoints_coverage.py
- [x] `GET /api/v2/marketplace/stats` — `api_marketplace_stats_v2`  ✓ test_marketplace_endpoints_coverage.py
- [x] `GET /marketplace` — `api_get_marketplace`  ✓ test_marketplace_endpoints_coverage.py
- [x] `POST /marketplace/bill/{job_id}` — `api_marketplace_bill`  ✓ test_marketplace_endpoints_coverage.py
- [x] `POST /marketplace/list` — `api_list_rig`
- [x] `GET /marketplace/search` — `api_marketplace_search`  ✓ test_marketplace_endpoints_coverage.py
- [x] `GET /marketplace/stats` — `api_marketplace_stats`  ✓ test_marketplace_endpoints_coverage.py
- [x] `DELETE /marketplace/{host_id}` — `api_unlist_rig`  ✓ test_marketplace_endpoints_coverage.py

### `routes/mfa.py` (0 untested)
- [x] `DELETE /api/auth/mfa/all` — `api_mfa_disable_all`  ✓ test_auth_endpoints_coverage.py
- [x] `POST /api/auth/mfa/backup-codes/regenerate` — `api_mfa_regenerate_backup_codes`  ✓ test_auth_endpoints_coverage.py
- [x] `GET /api/auth/mfa/methods` — `api_mfa_list_methods`
- [x] `POST /api/auth/mfa/passkey/authenticate-complete` — `api_mfa_passkey_authenticate_complete`  ✓ test_auth_endpoints_coverage.py
- [x] `POST /api/auth/mfa/passkey/authenticate-options` — `api_mfa_passkey_authenticate_options`  ✓ test_auth_endpoints_coverage.py
- [x] `POST /api/auth/mfa/passkey/delete` — `api_mfa_passkey_delete`  ✓ test_auth_endpoints_coverage.py
- [x] `POST /api/auth/mfa/passkey/register-complete` — `api_mfa_passkey_register_complete`  ✓ test_auth_endpoints_coverage.py
- [x] `POST /api/auth/mfa/passkey/register-options` — `api_mfa_passkey_register_options`  ✓ test_auth_endpoints_coverage.py
- [x] `DELETE /api/auth/mfa/sms` — `api_mfa_sms_disable`  ✓ test_auth_endpoints_coverage.py
- [x] `POST /api/auth/mfa/sms/send` — `api_mfa_sms_send_login`  ✓ test_auth_endpoints_coverage.py
- [x] `POST /api/auth/mfa/sms/setup` — `api_mfa_sms_setup`  ✓ test_auth_endpoints_coverage.py
- [x] `POST /api/auth/mfa/sms/verify` — `api_mfa_sms_verify`  ✓ test_auth_endpoints_coverage.py
- [x] `DELETE /api/auth/mfa/totp` — `api_mfa_totp_disable`  ✓ test_auth_endpoints_coverage.py
- [x] `POST /api/auth/mfa/totp/setup` — `api_mfa_totp_setup`  ✓ test_auth_endpoints_coverage.py
- [x] `POST /api/auth/mfa/totp/verify` — `api_mfa_totp_verify`  ✓ test_auth_endpoints_coverage.py
- [x] `POST /api/auth/mfa/verify` — `api_mfa_verify_login`  ✓ test_auth_endpoints_coverage.py

### `routes/notifications.py` (0 untested)
- [x] `GET /api/notifications` — `api_list_notifications`  ✓ test_notifications_endpoints_coverage.py
- [x] `GET /api/notifications/push/subscription` — `api_get_push_subscription_status`  ✓ test_notifications_endpoints_coverage.py
- [x] `POST /api/notifications/push/subscription` — `api_upsert_push_subscription`  ✓ test_notifications_endpoints_coverage.py
- [x] `DELETE /api/notifications/push/subscription` — `api_delete_push_subscription`  ✓ test_notifications_endpoints_coverage.py
- [x] `POST /api/notifications/read-all` — `api_mark_all_read`  ✓ test_notifications_endpoints_coverage.py
- [x] `GET /api/notifications/unread-count` — `api_notification_unread_count`  ✓ test_notifications_endpoints_coverage.py
- [x] `DELETE /api/notifications/{notification_id}` — `api_delete_notification`  ✓ test_notifications_endpoints_coverage.py
- [x] `POST /api/notifications/{notification_id}/read` — `api_mark_notification_read`  ✓ test_notifications_endpoints_coverage.py

### `routes/platform.py` (0 untested)
- [x] `GET /api/v2/platform/attestation-schema` — `api_attestation_schema`
- [x] `GET /api/v2/platform/capacity-forecast/{endpoint_id}` — `api_capacity_forecast`  ✓ test_untested_endpoints_coverage.py
- [x] `GET /api/v2/platform/h100-partner` — `api_h100_partner`
- [x] `GET /api/v2/platform/ops-plan` — `api_platform_ops_plan`
- [x] `GET /api/v2/platform/scip-alignment` — `api_scip_alignment`
- [x] `GET /api/v2/platform/scip-loi` — `api_scip_partner_loi`
- [x] `POST /api/v2/serverless/should-i-run-this` — `api_should_i_run_pel_job`

### `routes/privacy.py` (0 untested)
- [x] `POST /api/privacy/config` — `api_save_privacy_config`  ✓ test_privacy_endpoints_coverage.py
- [x] `GET /api/privacy/config/{org_id}` — `api_get_privacy_config`  ✓ test_privacy_endpoints_coverage.py
- [x] `POST /api/privacy/consent` — `api_record_consent`  ✓ test_privacy_endpoints_coverage.py
- [x] `GET /api/privacy/consent/{entity_id}` — `api_get_consents`  ✓ test_privacy_endpoints_coverage.py
- [x] `DELETE /api/privacy/consent/{entity_id}/{consent_type}` — `api_revoke_consent`  ✓ test_privacy_endpoints_coverage.py
- [x] `POST /api/privacy/purge-expired` — `api_purge_expired`  ✓ test_privacy_endpoints_coverage.py
- [x] `GET /api/privacy/retention-policies` — `api_retention_policies`  ✓ test_privacy_endpoints_coverage.py
- [x] `GET /api/privacy/retention-summary` — `api_retention_summary`  ✓ test_privacy_endpoints_coverage.py
- [x] `POST /api/v2/privacy/consent` — `api_privacy_record_consent`  ✓ test_privacy_endpoints_coverage.py
- [x] `DELETE /api/v2/privacy/consent/{purpose}` — `api_privacy_withdraw_consent`  ✓ test_privacy_endpoints_coverage.py
- [x] `GET /api/v2/privacy/consents` — `api_privacy_list_consents`  ✓ test_privacy_endpoints_coverage.py
- [x] `POST /api/v2/privacy/erase` — `api_privacy_right_to_erasure`  ✓ test_privacy_endpoints_coverage.py
- [x] `GET /api/v2/privacy/erase/{request_id}` — `api_privacy_erasure_status`

### `routes/providers.py` (0 untested)
- [x] `GET /api/providers` — `api_list_providers`  ✓ test_compliance_endpoints_coverage.py
- [x] `POST /api/providers/register` — `api_register_provider`  ✓ test_compliance_endpoints_coverage.py
- [x] `POST /api/providers/webhook` — `api_stripe_webhook`  ✓ test_providers_endpoints_coverage.py
- [x] `GET /api/providers/{provider_id}` — `api_get_provider`  ✓ test_compliance_endpoints_coverage.py
- [x] `POST /api/providers/{provider_id}/abandon-onboarding` — `api_abandon_onboarding`  ✓ test_compliance_endpoints_coverage.py
- [x] `POST /api/providers/{provider_id}/account-session` — `api_provider_account_session`  ✓ test_compliance_endpoints_coverage.py
- [x] `GET /api/providers/{provider_id}/earnings` — `api_provider_earnings`  ✓ test_compliance_endpoints_coverage.py
- [x] `POST /api/providers/{provider_id}/incorporation` — `api_upload_incorporation`  ✓ test_compliance_endpoints_coverage.py
- [x] `POST /api/providers/{provider_id}/payout` — `api_provider_payout`  ✓ test_compliance_endpoints_coverage.py
- [x] `GET /api/providers/{provider_id}/paypal` — `api_provider_paypal_status`  ✓ test_compliance_endpoints_coverage.py
- [x] `POST /api/providers/{provider_id}/paypal/disconnect` — `api_provider_paypal_disconnect`  ✓ test_compliance_endpoints_coverage.py
- [x] `POST /api/providers/{provider_id}/paypal/onboard` — `api_provider_paypal_onboard`  ✓ test_compliance_endpoints_coverage.py
- [x] `POST /api/providers/{provider_id}/paypal/refresh` — `api_provider_paypal_refresh`  ✓ test_compliance_endpoints_coverage.py
- [x] `POST /api/providers/{provider_id}/resume-onboarding` — `api_resume_onboarding`  ✓ test_compliance_endpoints_coverage.py
- [x] `POST /api/providers/{provider_id}/stripe/disconnect` — `api_provider_stripe_disconnect`  ✓ test_compliance_endpoints_coverage.py

### `routes/reputation.py` (0 untested)
- [x] `GET /api/reputation/leaderboard` — `api_reputation_leaderboard`
- [x] `GET /api/reputation/me` — `api_reputation_me`  ✓ test_reputation_endpoints_coverage.py
- [x] `POST /api/reputation/me/claim` — `api_reputation_claim`
- [x] `GET /api/reputation/me/journey` — `api_reputation_journey`  ✓ test_untested_endpoints_coverage.py
- [x] `POST /api/reputation/verify` — `api_grant_verification`
- [x] `GET /api/reputation/{entity_id}` — `api_get_reputation`  ✓ test_reputation_endpoints_coverage.py
- [x] `GET /api/reputation/{entity_id}/breakdown` — `api_reputation_breakdown`  ✓ test_reputation_endpoints_coverage.py
- [x] `GET /api/reputation/{entity_id}/history` — `api_reputation_history`  ✓ test_reputation_endpoints_coverage.py
- [x] `GET /api/trust-tiers` — `api_trust_tiers`

### `routes/serverless.py` (0 untested)
- [x] `POST /api/v1/serverless/endpoint-plans` — `api_v1_serverless_endpoint_plan`
- [x] `POST /api/v1/serverless/endpoint-plans/{plan_id}/execute` — `api_v1_execute_serverless_endpoint_plan`
- [x] `GET /api/v2/inference/endpoints` — `api_inference_compat_endpoint_health`  ✓ test_untested_endpoints_coverage.py
- [x] `POST /api/v2/inference/endpoints` — `api_inference_compat_endpoint_usage`  ✓ test_untested_endpoints_coverage.py
- [x] `GET /api/v2/serverless/batches/{batch_id}` — `api_serverless_get_batch`  ✓ test_untested_endpoints_coverage.py
- [x] `GET /api/v2/serverless/enabled` — `api_serverless_enabled`
- [x] `POST /api/v2/serverless/endpoints` — `api_serverless_create_endpoint`
- [x] `GET /api/v2/serverless/endpoints` — `api_serverless_list_endpoints`
- [x] `GET /api/v2/serverless/endpoints/{endpoint_id}` — `api_serverless_get_endpoint`
- [x] `PATCH /api/v2/serverless/endpoints/{endpoint_id}` — `api_serverless_patch_endpoint`
- [x] `DELETE /api/v2/serverless/endpoints/{endpoint_id}` — `api_serverless_delete_endpoint`
- [x] `POST /api/v2/serverless/endpoints/{endpoint_id}/batches` — `api_serverless_create_batch`
- [x] `GET /api/v2/serverless/endpoints/{endpoint_id}/health` — `api_serverless_endpoint_health`
- [x] `GET /api/v2/serverless/endpoints/{endpoint_id}/jobs` — `api_serverless_list_jobs`
- [x] `GET /api/v2/serverless/endpoints/{endpoint_id}/jobs/{job_id}` — `api_serverless_dashboard_job_status`
- [x] `POST /api/v2/serverless/endpoints/{endpoint_id}/jobs/{job_id}/cancel` — `api_serverless_dashboard_cancel_job`
- [x] `GET /api/v2/serverless/endpoints/{endpoint_id}/jobs/{job_id}/stream` — `api_serverless_dashboard_job_stream`
- [x] `POST /api/v2/serverless/endpoints/{endpoint_id}/keys` — `api_serverless_create_key`
- [x] `GET /api/v2/serverless/endpoints/{endpoint_id}/keys` — `api_serverless_list_keys`
- [x] `DELETE /api/v2/serverless/endpoints/{endpoint_id}/keys/{key_id}` — `api_serverless_revoke_key`
- [x] `GET /api/v2/serverless/endpoints/{endpoint_id}/metrics` — `api_serverless_endpoint_metrics`
- [x] `POST /api/v2/serverless/endpoints/{endpoint_id}/test/openai/v1/chat/completions` — `api_serverless_test_openai_chat`
- [x] `POST /api/v2/serverless/endpoints/{endpoint_id}/test/run` — `api_serverless_test_run`
- [x] `POST /api/v2/serverless/endpoints/{endpoint_id}/test/runsync` — `api_serverless_test_runsync`
- [x] `GET /api/v2/serverless/endpoints/{endpoint_id}/usage` — `api_serverless_endpoint_usage`
- [x] `POST /api/v2/serverless/endpoints/{endpoint_id}/warm` — `api_serverless_warm_endpoint`
- [x] `GET /api/v2/serverless/endpoints/{endpoint_id}/workers` — `api_serverless_list_workers`
- [x] `GET /api/v2/serverless/endpoints/{endpoint_id}/workers/{worker_id}/logs` — `api_serverless_worker_logs`
- [x] `GET /api/v2/serverless/endpoints/{endpoint_id}/workers/{worker_id}/logs/stream` — `api_serverless_worker_logs_stream`
- [x] `GET /api/v2/serverless/endpoints/{endpoint_id}/workers/{worker_id}/telemetry` — `api_serverless_worker_telemetry`
- [x] `POST /api/v2/serverless/github/resolve` — `api_serverless_github_resolve`  ✓ test_untested_endpoints_coverage.py
- [x] `GET /api/v2/serverless/preset-token-pricing` — `api_preset_token_pricing`  ✓ test_untested_endpoints_coverage.py
- [x] `POST /api/v2/serverless/workers/{worker_id}/exited` — `api_serverless_worker_job_event`
- [x] `POST /api/v2/serverless/workers/{worker_id}/heartbeat` — `api_serverless_worker_complete_job`
- [x] `POST /api/v2/serverless/workers/{worker_id}/jobs/claim` — `api_inference_compat_list_endpoints`
- [x] `GET /api/v2/serverless/workers/{worker_id}/jobs/{job_id}` — `api_inference_compat_create_endpoint`
- [x] `POST /api/v2/serverless/workers/{worker_id}/jobs/{job_id}/complete` — `api_inference_compat_get_endpoint`
- [x] `POST /api/v2/serverless/workers/{worker_id}/jobs/{job_id}/events` — `api_inference_compat_delete_endpoint`
- [x] `POST /api/v2/serverless/workers/{worker_id}/ready` — `api_serverless_worker_get_job`
- [x] `POST /v1/serverless/{endpoint_id}/cancel/{job_id}` — `_stream`  ✓ test_chat_endpoints_coverage.py
- [x] `POST /v1/serverless/{endpoint_id}/openai/v1/chat/completions` — `api_serverless_openai_models`
- [x] `POST /v1/serverless/{endpoint_id}/openai/v1/embeddings` — `api_serverless_worker_ready`
- [x] `GET /v1/serverless/{endpoint_id}/openai/v1/models` — `api_serverless_worker_exited`
- [x] `POST /v1/serverless/{endpoint_id}/run` — `api_serverless_run`
- [x] `POST /v1/serverless/{endpoint_id}/runsync` — `api_serverless_job_status`
- [x] `GET /v1/serverless/{endpoint_id}/status/{job_id}` — `_on_disconnect`
- [x] `GET /v1/serverless/{endpoint_id}/stream/{job_id}` — `api_serverless_cancel_job`
- [x] `POST /v1/serverless/{endpoint_id}/{endpoint_slug}/cancel/{job_id}` — `api_serverless_openai_embeddings`
- [x] `POST /v1/serverless/{endpoint_id}/{endpoint_slug}/openai/v1/chat/completions` — `_require_worker_callback`
- [x] `POST /v1/serverless/{endpoint_id}/{endpoint_slug}/openai/v1/embeddings` — `api_serverless_worker_heartbeat`
- [x] `GET /v1/serverless/{endpoint_id}/{endpoint_slug}/openai/v1/models` — `api_serverless_worker_claim_job`
- [x] `POST /v1/serverless/{endpoint_id}/{endpoint_slug}/run` — `api_serverless_runsync`
- [x] `POST /v1/serverless/{endpoint_id}/{endpoint_slug}/runsync` — `api_serverless_job_stream`
- [x] `GET /v1/serverless/{endpoint_id}/{endpoint_slug}/status/{job_id}` — `_gen`  ✓ test_health_endpoints_coverage.py
- [x] `GET /v1/serverless/{endpoint_id}/{endpoint_slug}/stream/{job_id}` — `api_serverless_openai_chat`

### `routes/sla.py` (0 untested)
- [x] `GET /api/sla/downtimes` — `api_sla_active_downtimes`
- [x] `POST /api/sla/enforce` — `api_sla_enforce`
- [x] `GET /api/sla/hosts-summary` — `api_sla_hosts_summary`
- [x] `GET /api/sla/targets` — `api_sla_targets`
- [x] `GET /api/sla/violations/{host_id}` — `api_sla_violations`  ✓ test_sla_endpoints_coverage.py
- [x] `GET /api/sla/{host_id}` — `api_sla_status`  ✓ test_sla_endpoints_coverage.py

### `routes/spot.py` (0 untested)
- [x] `GET /api/pricing/spot-enabled` — `api_spot_enabled`
- [x] `GET /api/pricing/spot-floor-suggestion` — `api_spot_floor_suggestion`  ✓ test_untested_endpoints_coverage.py
- [x] `GET /spot-prices` — `api_spot_prices`  ✓ test_marketplace_endpoints_coverage.py
- [x] `POST /spot-prices/update` — `api_update_spot_prices`
- [x] `POST /spot/preemption-cycle` — `api_preemption_cycle`

### `routes/ssh.py` (0 untested)
- [x] `POST /api/ssh/keys` — `api_add_ssh_key`
- [x] `GET /api/ssh/keys` — `api_list_ssh_keys`
- [x] `DELETE /api/ssh/keys/{key_id}` — `api_delete_ssh_key`

### `routes/stripe_connect_v2.py` (0 untested)
- [x] `POST /api/connect/accounts` — `create_connected_account`  ✓ test_stripe_connect_v2_endpoints_coverage.py
- [x] `GET /api/connect/accounts` — `list_connected_accounts`  ✓ test_stripe_connect_v2_endpoints_coverage.py
- [x] `GET /api/connect/accounts/{account_id}/onboarding-link` — `create_onboarding_link`  ✓ test_stripe_connect_v2_endpoints_coverage.py
- [x] `GET /api/connect/accounts/{account_id}/status` — `get_account_status`  ✓ test_stripe_connect_v2_endpoints_coverage.py
- [x] `POST /api/connect/checkout` — `create_checkout_session`  ✓ test_stripe_connect_v2_endpoints_coverage.py
- [x] `POST /api/connect/products` — `create_product`  ✓ test_stripe_connect_v2_endpoints_coverage.py
- [x] `GET /api/connect/products` — `list_products`  ✓ test_stripe_connect_v2_endpoints_coverage.py
- [x] `POST /api/connect/webhooks` — `handle_thin_webhook`  ✓ test_stripe_connect_v2_endpoints_coverage.py
- [x] `GET /connect/dashboard` — `connect_dashboard_page`  ✓ test_stripe_connect_v2_endpoints_coverage.py
- [x] `GET /connect/storefront` — `storefront_page`  ✓ test_stripe_connect_v2_endpoints_coverage.py
- [x] `GET /connect/success` — `success_page`  ✓ test_stripe_connect_v2_endpoints_coverage.py

### `routes/teams.py` (0 untested)
- [x] `POST /api/teams` — `api_create_team`  ✓ test_teams_endpoints_coverage.py
- [x] `PATCH /api/teams/active` — `api_set_active_team`  ✓ test_teams_endpoints_coverage.py
- [x] `GET /api/teams/invite/{token}` — `api_accept_team_invite`  ✓ test_teams_endpoints_coverage.py
- [x] `POST /api/teams/invite/{token}/accept` — `api_accept_invite_authenticated`  ✓ test_teams_endpoints_coverage.py
- [x] `GET /api/teams/me` — `api_my_teams`  ✓ test_teams_endpoints_coverage.py
- [x] `GET /api/teams/{team_id}` — `api_get_team`  ✓ test_teams_endpoints_coverage.py
- [x] `DELETE /api/teams/{team_id}` — `api_delete_team`  ✓ test_teams_endpoints_coverage.py
- [x] `POST /api/teams/{team_id}/members` — `api_add_team_member`  ✓ test_teams_endpoints_coverage.py
- [x] `DELETE /api/teams/{team_id}/members/{email}` — `api_remove_team_member`  ✓ test_teams_endpoints_coverage.py
- [x] `PATCH /api/teams/{team_id}/members/{email}` — `api_update_team_member_role`  ✓ test_teams_endpoints_coverage.py

### `routes/terminal.py` (0 untested)
- [x] `POST /api/terminal/ticket` — `api_terminal_ticket`

### `routes/transparency.py` (0 untested)
- [x] `POST /api/transparency/legal-request` — `api_record_legal_request`  ✓ test_transparency_endpoints_coverage.py
- [x] `POST /api/transparency/legal-request/{request_id}/respond` — `api_respond_legal_request`  ✓ test_transparency_endpoints_coverage.py
- [x] `GET /api/transparency/report` — `api_transparency_report`  ✓ test_transparency_endpoints_coverage.py

### `routes/verification.py` (0 untested)
- [x] `POST /agent/verify` — `api_agent_verify`  ✓ test_verification_endpoints_coverage.py
- [x] `GET /api/verified-hosts` — `api_verified_hosts`  ✓ test_verification_endpoints_coverage.py
- [x] `POST /api/verify/{host_id}` — `api_verify_host`  ✓ test_verification_endpoints_coverage.py
- [x] `POST /api/verify/{host_id}/approve` — `api_admin_approve_host`  ✓ test_verification_endpoints_coverage.py
- [x] `POST /api/verify/{host_id}/reject` — `api_admin_reject_host`  ✓ test_verification_endpoints_coverage.py
- [x] `GET /api/verify/{host_id}/status` — `api_verification_status`  ✓ test_verification_endpoints_coverage.py

### `routes/volumes.py` (0 untested)
- [x] `POST /api/v1/promotions/{promotion_id}/files` — `api_promotion_file_result`  ✓ test_untested_endpoints_coverage.py
- [x] `GET /api/v1/promotions/{promotion_id}/manifest` — `api_promotion_manifest_for_agent`  ✓ test_untested_endpoints_coverage.py
- [x] `POST /api/v1/promotions/{promotion_id}/result` — `api_promotion_result_from_agent`  ✓ test_untested_endpoints_coverage.py
- [x] `POST /api/v2/admin/volumes/reopen-encrypted` — `api_admin_reopen_encrypted_volumes`
- [x] `POST /api/v2/volumes` — `api_volume_create`  ✓ test_volumes_endpoints_coverage.py
- [x] `GET /api/v2/volumes` — `api_volume_list`  ✓ test_volumes_endpoints_coverage.py
- [x] `GET /api/v2/volumes/available` — `api_volumes_available`
- [x] `GET /api/v2/volumes/{volume_id}` — `api_volume_get`  ✓ test_volumes_endpoints_coverage.py
- [x] `PATCH /api/v2/volumes/{volume_id}` — `api_volume_rename`  ✓ test_volumes_endpoints_coverage.py
- [x] `DELETE /api/v2/volumes/{volume_id}` — `api_volume_delete`  ✓ test_volumes_endpoints_coverage.py
- [x] `POST /api/v2/volumes/{volume_id}/attach` — `api_volume_attach`  ✓ test_volumes_endpoints_coverage.py
- [x] `POST /api/v2/volumes/{volume_id}/detach` — `api_volume_detach`  ✓ test_volumes_endpoints_coverage.py
- [x] `POST /api/v2/volumes/{volume_id}/promotions` — `api_volume_promotion_create`  ✓ test_volumes_endpoints_coverage.py
- [x] `GET /api/v2/volumes/{volume_id}/promotions/preview` — `api_volume_promotion_preview`  ✓ test_volumes_endpoints_coverage.py
- [x] `GET /api/v2/volumes/{volume_id}/promotions/{promotion_id}` — `api_volume_promotion_get`  ✓ test_volumes_endpoints_coverage.py
- [x] `POST /api/v2/volumes/{volume_id}/retry` — `api_volume_retry_provision`  ✓ test_volumes_endpoints_coverage.py
- [x] `POST /api/v2/volumes/{volume_id}/snapshots` — `api_volume_snapshot_create`  ✓ test_volumes_endpoints_coverage.py
- [x] `GET /api/v2/volumes/{volume_id}/snapshots` — `api_volume_snapshot_list`  ✓ test_volumes_endpoints_coverage.py
- [x] `DELETE /api/v2/volumes/{volume_id}/snapshots/{snapshot_id}` — `api_volume_snapshot_delete`  ✓ test_volumes_endpoints_coverage.py
- [x] `POST /api/v2/volumes/{volume_id}/snapshots/{snapshot_id}/restore` — `api_volume_snapshot_restore`  ✓ test_volumes_endpoints_coverage.py

## CLI commands (0 untested)

- [x] `autoscale` — `cmd_autoscale`
- [x] `bill` — `cmd_bill`
- [x] `build` — `cmd_build`
- [x] `builds` — `cmd_builds`
- [x] `cancel` — `cmd_cancel`
- [x] `config` — `cmd_config`
- [x] `deposit` — `cmd_deposit`
- [x] `failover` — `cmd_failover`
- [x] `health-start` — `cmd_health_start`
- [x] `host-accept` — `cmd_host_accept`
- [x] `host-add` — `cmd_host_add`
- [x] `host-add-located` — `cmd_host_add_located`
- [x] `host-profile` — `cmd_host_profile`
- [x] `host-rm` — `cmd_host_rm`
- [x] `hosts` — `cmd_hosts`
- [x] `invoice` — `cmd_invoice`
- [x] `job` — `cmd_job`
- [x] `jobs` — `cmd_jobs`
- [x] `leaderboard` — `cmd_leaderboard`
- [x] `login` — `cmd_login`
- [x] `logout` — `cmd_logout`
- [x] `market` — `cmd_market`
- [x] `market-list` — `cmd_market_list`
- [x] `market-stats` — `cmd_market_stats`
- [x] `market-unlist` — `cmd_market_unlist`
- [x] `ping` — `cmd_ping`
- [x] `pool` — `cmd_pool`
- [x] `pool-add` — `cmd_pool_add`
- [x] `pool-rm` — `cmd_pool_rm`
- [x] `process` — `cmd_process`
- [x] `provider-info` — `cmd_provider_info`
- [x] `provider-register` — `cmd_provider_register`
- [x] `reputation` — `cmd_reputation`
- [x] `requeue` — `cmd_requeue`
- [x] `revenue` — `cmd_revenue`
- [x] `run` — `cmd_run`
- [x] `serve` — `cmd_serve`
- [x] `sla` — `cmd_sla`
- [x] `slurm-cancel` — `cmd_slurm_cancel`
- [x] `slurm-status` — `cmd_slurm_status`
- [x] `slurm-submit` — `cmd_slurm_submit`
- [x] `ssh-keygen` — `cmd_ssh_keygen`
- [x] `ssh-pubkey` — `cmd_ssh_pubkey`
- [x] `tiers` — `cmd_tiers`
- [x] `token-gen` — `cmd_token_gen`
- [x] `verify` — `cmd_verify`
- [x] `wallet` — `cmd_wallet`
- [x] `whoami` — `cmd_whoami`
