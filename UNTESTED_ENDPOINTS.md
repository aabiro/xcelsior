# Untested endpoints — coverage-gap worklist

_Generated 2026-06-03 by a coverage heuristic: a route/CLI command counts as untested when **neither** its path (with `{params}` genericized) **nor** its handler function name appears anywhere under `tests/`._

**170 of 375 routes (45%)** and **16 of 51 CLI commands** have no test signal.

Workflow per item: write a `TestClient` (or CLI) test → if it works, tick the box; if it 500s/throws, fix-or-delete then tick. Caveat: a few may be exercised transitively; confirm with the test.

## Routes (170 untested)

### `routes/billing.py` (21)
- [x] `DELETE /api/billing/payment-methods/{payment_method_id}` — `api_billing_detach_payment_method`  ✓ test_billing_endpoints_coverage.py
- [x] `GET /api/billing/crypto/deposit/{deposit_id}` — `api_crypto_deposit_status`  ✓ test_billing_endpoints_coverage.py
- [x] `GET /api/billing/crypto/rate` — `api_crypto_rate`  ✓ test_billing_endpoints_coverage.py
- [x] `GET /api/billing/invoice/{customer_id}` — `api_generate_invoice`  ✓ test_billing_endpoints_coverage.py
- [x] `GET /api/billing/invoice/{customer_id}/download` — `api_download_invoice`  ✓ test_billing_endpoints_coverage.py
- [x] `GET /api/billing/lightning/deposit/{deposit_id}` — `api_ln_check_deposit`  ✓ test_billing_endpoints_coverage.py
- [x] `GET /api/billing/lightning/enabled` — `api_ln_enabled`  ✓ test_billing_endpoints_coverage.py
- [x] `GET /api/billing/lightning/rate` — `api_ln_rate`  ✓ test_billing_endpoints_coverage.py
- [x] `GET /api/billing/payment-methods` — `api_billing_list_payment_methods`  ✓ test_billing_endpoints_coverage.py
- [x] `GET /api/billing/paypal/enabled` — `api_paypal_enabled`  ✓ test_billing_endpoints_coverage.py
- [x] `GET /api/billing/usage/{customer_id}` — `api_usage_summary`  ✓ test_billing_endpoints_coverage.py
- [x] `GET /api/pricing/models` — `api_pricing_models`  ✓ test_billing_endpoints_coverage.py
- [x] `GET /api/pricing/rates` — `api_pricing_rates`  ✓ test_billing_endpoints_coverage.py
- [x] `GET /api/pricing/reservations` — `api_list_reservations`  ✓ test_billing_endpoints_coverage.py
- [x] `GET /api/v2/billing/auto-topup` — `api_billing_get_topup`  ✓ test_billing_endpoints_coverage.py
- [x] `POST /api/billing/crypto/deposit` — `api_crypto_deposit`  ✓ test_billing_endpoints_coverage.py
- [x] `POST /api/billing/crypto/refresh/{deposit_id}` — `api_crypto_refresh`  ✓ test_billing_endpoints_coverage.py
- [x] `POST /api/billing/lightning/deposit` — `api_ln_create_deposit`  ✓ test_billing_endpoints_coverage.py
- [x] `POST /api/billing/payment-intent` — `api_create_payment_intent`  ✓ test_billing_endpoints_coverage.py
- [x] `POST /api/billing/setup-intent` — `api_billing_setup_intent`  ✓ test_billing_endpoints_coverage.py
- [x] `POST /api/v2/billing/auto-topup` — `api_billing_configure_topup`  ✓ test_billing_endpoints_coverage.py

### `routes/health.py` (17)
- [x] `GET /_internal/legacy-auth/verify` — `api_auth_verify_page`  ✓ test_health_endpoints_coverage.py
- [x] `GET /alerts/config` — `api_get_alert_config`  ✓ test_health_endpoints_coverage.py
- [x] `GET /api/alerts/config` — `api_get_alert_config_alias`  ✓ test_health_endpoints_coverage.py
- [x] `GET /api/slurm/instances` — `api_slurm_list_instances`  ✓ test_health_endpoints_coverage.py
- [x] `GET /api/ssh/pubkey` — `api_get_pubkey`  ✓ test_health_endpoints_coverage.py
- [x] `GET /builds` — `api_list_builds`  ✓ test_health_endpoints_coverage.py
- [x] `GET /legacy/{path:path}` — `legacy_dashboard`  ✓ test_health_endpoints_coverage.py
- [x] `GET /llms.txt` — `api_llms_txt`  ✓ test_health_endpoints_coverage.py
- [x] `GET /ssh/pubkey` — `api_get_pubkey`  ✓ test_health_endpoints_coverage.py
- [x] `POST /_internal/legacy-auth/device` — `api_auth_device_code`  ✓ test_health_endpoints_coverage.py
- [x] `POST /_internal/legacy-auth/token` — `api_auth_device_token`  ✓ test_health_endpoints_coverage.py
- [x] `POST /_internal/legacy-auth/verify` — `api_auth_verify_device`  ✓ test_health_endpoints_coverage.py
- [x] `POST /build` — `api_build_image`  ✓ test_health_endpoints_coverage.py
- [x] `POST /build/{model}/dockerfile` — `api_generate_dockerfile`  ✓ test_health_endpoints_coverage.py
- [x] `POST /ssh/keygen` — `api_generate_ssh_key`  ✓ test_health_endpoints_coverage.py
- [x] `PUT /alerts/config` — `api_set_alert_config`  ✓ test_health_endpoints_coverage.py
- [x] `PUT /api/alerts/config` — `api_set_alert_config_alias`  ✓ test_health_endpoints_coverage.py

### `routes/mfa.py` (15)
- [x] `DELETE /api/auth/mfa/all` — `api_mfa_disable_all`  ✓ test_auth_endpoints_coverage.py
- [x] `DELETE /api/auth/mfa/sms` — `api_mfa_sms_disable`  ✓ test_auth_endpoints_coverage.py
- [x] `DELETE /api/auth/mfa/totp` — `api_mfa_totp_disable`  ✓ test_auth_endpoints_coverage.py
- [x] `POST /api/auth/mfa/backup-codes/regenerate` — `api_mfa_regenerate_backup_codes`  ✓ test_auth_endpoints_coverage.py
- [x] `POST /api/auth/mfa/passkey/authenticate-complete` — `api_mfa_passkey_authenticate_complete`  ✓ test_auth_endpoints_coverage.py
- [x] `POST /api/auth/mfa/passkey/authenticate-options` — `api_mfa_passkey_authenticate_options`  ✓ test_auth_endpoints_coverage.py
- [x] `POST /api/auth/mfa/passkey/delete` — `api_mfa_passkey_delete`  ✓ test_auth_endpoints_coverage.py
- [x] `POST /api/auth/mfa/passkey/register-complete` — `api_mfa_passkey_register_complete`  ✓ test_auth_endpoints_coverage.py
- [x] `POST /api/auth/mfa/passkey/register-options` — `api_mfa_passkey_register_options`  ✓ test_auth_endpoints_coverage.py
- [x] `POST /api/auth/mfa/sms/send` — `api_mfa_sms_send_login`  ✓ test_auth_endpoints_coverage.py
- [x] `POST /api/auth/mfa/sms/setup` — `api_mfa_sms_setup`  ✓ test_auth_endpoints_coverage.py
- [x] `POST /api/auth/mfa/sms/verify` — `api_mfa_sms_verify`  ✓ test_auth_endpoints_coverage.py
- [x] `POST /api/auth/mfa/totp/setup` — `api_mfa_totp_setup`  ✓ test_auth_endpoints_coverage.py
- [x] `POST /api/auth/mfa/totp/verify` — `api_mfa_totp_verify`  ✓ test_auth_endpoints_coverage.py
- [x] `POST /api/auth/mfa/verify` — `api_mfa_verify_login`  ✓ test_auth_endpoints_coverage.py

### `routes/admin.py` (13)
- [x] `DELETE /api/admin/teams/{team_id}/members/{email}` — `api_admin_remove_team_member`  ✓ test_admin_endpoints_coverage.py
- [x] `GET /api/admin/activity` — `api_admin_activity`  ✓ test_admin_endpoints_coverage.py
- [x] `GET /api/admin/ai-conversations` — `api_admin_ai_conversations`  ✓ test_admin_endpoints_coverage.py
- [x] `GET /api/admin/ai-stats` — `api_admin_ai_stats`  ✓ test_admin_endpoints_coverage.py
- [x] `GET /api/admin/infrastructure` — `api_admin_infrastructure`  ✓ test_admin_endpoints_coverage.py
- [x] `GET /api/admin/overview` — `api_admin_overview`  ✓ test_admin_endpoints_coverage.py
- [x] `GET /api/admin/revenue` — `api_admin_revenue`  ✓ test_admin_endpoints_coverage.py
- [x] `GET /api/admin/teams` — `api_admin_teams`  ✓ test_admin_endpoints_coverage.py
- [x] `GET /api/admin/users` — `api_admin_users`  ✓ test_admin_endpoints_coverage.py
- [x] `GET /api/admin/verification-queue` — `api_admin_verification_queue`  ✓ test_admin_endpoints_coverage.py
- [x] `POST /api/admin/agent/rollout` — `api_admin_agent_rollout`  ✓ test_admin_endpoints_coverage.py
- [x] `POST /api/admin/users/{email}/role` — `api_admin_set_user_role`  ✓ test_admin_endpoints_coverage.py
- [x] `POST /api/admin/users/{email}/toggle-admin` — `api_admin_toggle_admin`  ✓ test_admin_endpoints_coverage.py

### `routes/stripe_connect_v2.py` (11)
- [ ] `GET /api/connect/accounts` — `list_connected_accounts`
- [ ] `GET /api/connect/accounts/{account_id}/onboarding-link` — `create_onboarding_link`
- [ ] `GET /api/connect/accounts/{account_id}/status` — `get_account_status`
- [ ] `GET /api/connect/products` — `list_products`
- [ ] `GET /connect/dashboard` — `connect_dashboard_page`
- [ ] `GET /connect/storefront` — `storefront_page`
- [ ] `GET /connect/success` — `success_page`
- [ ] `POST /api/connect/accounts` — `create_connected_account`
- [ ] `POST /api/connect/checkout` — `create_checkout_session`
- [ ] `POST /api/connect/products` — `create_product`
- [ ] `POST /api/connect/webhooks` — `handle_thin_webhook`

### `routes/auth.py` (9)
- [x] `DELETE /api/auth/sessions/{token_prefix}` — `api_auth_revoke_session`  ✓ test_auth_endpoints_coverage.py
- [x] `GET /api/auth/oauth/{provider}/callback` — `api_auth_oauth_callback`  ✓ test_auth_endpoints_coverage.py
- [x] `GET /api/auth/sessions` — `api_auth_list_sessions`  ✓ test_auth_endpoints_coverage.py
- [x] `GET /api/users/me/preferences` — `api_get_user_preferences`  ✓ test_auth_endpoints_coverage.py
- [x] `POST /api/auth/logout` — `api_auth_logout`  ✓ test_auth_endpoints_coverage.py
- [x] `POST /api/auth/resend-verification` — `api_auth_resend_verification`  ✓ test_auth_endpoints_coverage.py
- [x] `POST /api/oauth/clients/{client_id}/rotate-secret` — `api_rotate_oauth_client_secret`  ✓ test_auth_endpoints_coverage.py
- [x] `POST /oauth/device/authorize` — `oauth_device_authorize`  ✓ test_auth_endpoints_coverage.py
- [x] `PUT /api/users/me/preferences` — `api_set_user_preferences`  ✓ test_auth_endpoints_coverage.py

### `routes/chat.py` (9)
- [x] `DELETE /api/ai/conversations/{conversation_id}` — `api_ai_delete_conversation`  ✓ test_chat_endpoints_coverage.py
- [x] `GET /api/ai/conversations` — `api_ai_list_conversations`  ✓ test_chat_endpoints_coverage.py
- [x] `GET /api/ai/conversations/{conversation_id}` — `api_ai_get_conversation`  ✓ test_chat_endpoints_coverage.py
- [x] `GET /api/ai/suggestions` — `api_ai_suggestions`  ✓ test_chat_endpoints_coverage.py
- [x] `GET /api/chat/conversations` — `api_chat_conversations`  ✓ test_chat_endpoints_coverage.py
- [x] `POST /api/ai/analytics` — `api_ai_analytics_chat`  ✓ test_chat_endpoints_coverage.py
- [x] `POST /api/ai/chat` — `api_ai_chat`  ✓ test_chat_endpoints_coverage.py
- [x] `POST /api/ai/confirm` — `api_ai_confirm`  ✓ test_chat_endpoints_coverage.py
- [x] `POST /api/chat/feedback` — `api_chat_feedback`  ✓ test_chat_endpoints_coverage.py

### `routes/providers.py` (9)
- [x] `GET /api/providers` — `api_list_providers`  ✓ test_providers_endpoints_coverage.py
- [x] `GET /api/providers/{provider_id}` — `api_get_provider`  ✓ test_providers_endpoints_coverage.py
- [x] `GET /api/providers/{provider_id}/earnings` — `api_provider_earnings`  ✓ test_providers_endpoints_coverage.py
- [x] `POST /api/providers/register` — `api_register_provider`  ✓ test_providers_endpoints_coverage.py
- [x] `POST /api/providers/webhook` — `api_stripe_webhook`  ✓ test_providers_endpoints_coverage.py
- [x] `POST /api/providers/{provider_id}/abandon-onboarding` — `api_abandon_onboarding`  ✓ test_providers_endpoints_coverage.py
- [x] `POST /api/providers/{provider_id}/incorporation` — `api_upload_incorporation`  ✓ test_providers_endpoints_coverage.py
- [x] `POST /api/providers/{provider_id}/payout` — `api_provider_payout`  ✓ test_providers_endpoints_coverage.py
- [x] `POST /api/providers/{provider_id}/resume-onboarding` — `api_resume_onboarding`  ✓ test_providers_endpoints_coverage.py

### `routes/instances.py` (7)
- [ ] `GET /api/images/templates` — `api_image_templates`
- [ ] `GET /instances/{job_id}/logs/stream` — `api_instance_log_stream`
- [ ] `PATCH /instance/{job_id}/name` — `api_rename_instance`
- [ ] `POST /admin/instances/{job_id}/reinject-shell` — `api_admin_reinject_shell`
- [ ] `POST /instances/{job_id}/lock` — `api_lock_instance`
- [ ] `POST /instances/{job_id}/reset` — `api_reset_instance`
- [ ] `POST /instances/{job_id}/unlock` — `api_unlock_instance`

### `routes/notifications.py` (7)
- [ ] `DELETE /api/notifications/push/subscription` — `api_delete_push_subscription`
- [ ] `DELETE /api/notifications/{notification_id}` — `api_delete_notification`
- [ ] `GET /api/notifications/push/subscription` — `api_get_push_subscription_status`
- [ ] `GET /api/notifications/unread-count` — `api_notification_unread_count`
- [ ] `POST /api/notifications/push/subscription` — `api_upsert_push_subscription`
- [ ] `POST /api/notifications/read-all` — `api_mark_all_read`
- [ ] `POST /api/notifications/{notification_id}/read` — `api_mark_notification_read`

### `routes/compliance.py` (6)
- [ ] `GET /api/billing/gst-threshold/{provider_id}` — `api_provider_gst_threshold`
- [ ] `GET /api/compliance/detect-province` — `api_detect_province`
- [ ] `GET /api/compliance/provinces` — `api_compliance_provinces`
- [ ] `GET /api/compliance/tax-rates` — `api_tax_rates`
- [ ] `GET /api/compliance/trust-tier-requirements` — `api_trust_tier_requirements`
- [ ] `POST /api/compliance/quebec-pia-check` — `api_quebec_pia_check`

### `routes/marketplace.py` (6)
- [ ] `GET /api/v2/marketplace/spot-prices` — `api_marketplace_spot_prices`
- [ ] `GET /api/v2/marketplace/spot-prices/{gpu_model}/history` — `api_marketplace_spot_history`
- [ ] `GET /api/v2/marketplace/stats` — `api_marketplace_stats_v2`
- [ ] `POST /api/v2/marketplace/release/{allocation_id}` — `api_marketplace_release`
- [ ] `POST /api/v2/marketplace/search` — `api_marketplace_search`
- [ ] `POST /marketplace/bill/{job_id}` — `api_marketplace_bill`

### `routes/inference.py` (5)
- [ ] `DELETE /api/v2/inference/endpoints/{endpoint_id}` — `api_inference_delete_endpoint`
- [ ] `GET /api/v2/inference/endpoints/{endpoint_id}` — `api_inference_get_endpoint`
- [ ] `GET /api/v2/inference/endpoints/{endpoint_id}/health` — `api_inference_endpoint_health`
- [ ] `GET /api/v2/inference/endpoints/{endpoint_id}/usage` — `api_inference_endpoint_usage`
- [ ] `POST /v1/chat/completions` — `api_openai_chat_completions`

### `routes/privacy.py` (5)
- [ ] `DELETE /api/v2/privacy/consent/{purpose}` — `api_privacy_withdraw_consent`
- [ ] `GET /api/privacy/config/{org_id}` — `api_get_privacy_config`
- [ ] `POST /api/privacy/config` — `api_save_privacy_config`
- [ ] `POST /api/privacy/purge-expired` — `api_purge_expired`
- [ ] `POST /api/v2/privacy/erase` — `api_privacy_right_to_erasure`

### `routes/verification.py` (5)
- [ ] `GET /api/verify/{host_id}/status` — `api_verification_status`
- [ ] `POST /agent/verify` — `api_agent_verify`
- [ ] `POST /api/verify/{host_id}` — `api_verify_host`
- [ ] `POST /api/verify/{host_id}/approve` — `api_admin_approve_host`
- [ ] `POST /api/verify/{host_id}/reject` — `api_admin_reject_host`

### `routes/events.py` (4)
- [ ] `GET /api/audit/instance/{job_id}` — `api_instance_audit_trail`
- [ ] `GET /api/audit/verify-chain` — `api_verify_event_chain`
- [ ] `GET /api/events/leases/{job_id}` — `api_get_lease`
- [ ] `GET /api/events/{entity_type}/{entity_id}` — `api_get_events`

### `routes/volumes.py` (4)
- [ ] `DELETE /api/v2/volumes/{volume_id}/snapshots/{snapshot_id}` — `api_volume_snapshot_delete`
- [ ] `GET /api/v2/volumes/{volume_id}/snapshots` — `api_volume_snapshot_list`
- [ ] `POST /api/v2/volumes/{volume_id}/snapshots` — `api_volume_snapshot_create`
- [ ] `POST /api/v2/volumes/{volume_id}/snapshots/{snapshot_id}/restore` — `api_volume_snapshot_restore`

### `routes/autoscale.py` (3)
- [ ] `POST /autoscale/cycle` — `api_autoscale_cycle`
- [ ] `POST /autoscale/down` — `api_autoscale_down`
- [ ] `POST /autoscale/up` — `api_autoscale_up`

### `routes/jurisdiction.py` (3)
- [ ] `POST /api/jurisdiction/hosts` — `api_jurisdiction_hosts`
- [ ] `POST /api/queue/process-sovereign` — `api_process_queue_sovereign`
- [ ] `POST /queue/process/ca` — `api_process_queue_ca`

### `routes/hosts.py` (2)
- [ ] `POST /api/hosts/register` — `api_register_host_web`
- [ ] `POST /hosts/check` — `api_check_hosts`

### `routes/teams.py` (2)
- [ ] `GET /api/teams/invite/{token}` — `api_accept_team_invite`
- [ ] `POST /api/teams/invite/{token}/accept` — `api_accept_invite_authenticated`

### `routes/transparency.py` (2)
- [ ] `POST /api/transparency/legal-request` — `api_record_legal_request`
- [ ] `POST /api/transparency/legal-request/{request_id}/respond` — `api_respond_legal_request`

### `routes/agent.py` (1)
- [ ] `POST /agent/ssh-status/{job_id}` — `api_agent_ssh_status`

### `routes/artifacts.py` (1)
- [ ] `POST /api/artifacts/download` — `api_request_download`

### `routes/cloudburst.py` (1)
- [ ] `GET /api/v2/burst/status` — `api_burst_status`

### `routes/reputation.py` (1)
- [ ] `GET /api/reputation/me` — `api_reputation_me`

### `routes/sla.py` (1)
- [ ] `GET /api/sla/violations/{host_id}` — `api_sla_violations`

## CLI commands (16 untested)

- [ ] `health-start` — `cmd_health_start`
- [ ] `host-accept` — `cmd_host_accept`
- [ ] `host-add` — `cmd_host_add`
- [ ] `host-add-ca` — `cmd_host_add_ca`
- [ ] `host-profile` — `cmd_host_profile`
- [ ] `host-rm` — `cmd_host_rm`
- [ ] `hosts-ca` — `cmd_hosts_ca`
- [ ] `market-stats` — `cmd_market_stats`
- [ ] `market-unlist` — `cmd_market_unlist`
- [ ] `pool-add` — `cmd_pool_add`
- [ ] `pool-rm` — `cmd_pool_rm`
- [ ] `provider-register` — `cmd_provider_register`
- [ ] `slurm-cancel` — `cmd_slurm_cancel`
- [ ] `slurm-submit` — `cmd_slurm_submit`
- [ ] `ssh-pubkey` — `cmd_ssh_pubkey`
- [ ] `token-gen` — `cmd_token_gen`
