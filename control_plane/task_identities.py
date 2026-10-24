"""Database authority required by each maintenance task's current code.

Tasks spanning several domains use the existing API writer. This is explicit
technical debt, not permission to widen the billing/reconciler roles. Partition
DDL moves to the separately credentialed, one-shot maintenance command.
"""

TASK_SERVICES = {
    "maintenance_heartbeat": "reconciler",
    "billing_cycle": "billing",
    "billing_control_cycle": "api",  # fenced lifecycle and volume housekeeping
    "stripe_meter_outbox": "billing",
    "provider_settlement_daily": "billing",
    "billing_meter_reconcile": "api",  # enforcing repair can write meters
    "job_reconciliation_recovery": "reconciler",
    "audit_checkpoint": "projector",
    "projection_fanout_prepare": "projector",
    "projection_deliver_audit_log": "projector",
    "projection_delivery_retention": "projector",
    "wallet_hold_expiry": "billing",
    "host_agent_token_expiry": "reconciler",
    "webhook_processor": "api",  # payment handlers can update identity
    "spot_updater": "api",
    "serverless_reconcile": "api",
    "serverless_webhook_retry": "api",
    "serverless_callback_credential_retention": "reconciler",
    "burst_evaluator": "api",
    "sla_credits": "api",
    "reverification_sweep": "api",
    "placement_observation_maintenance": "reconciler",
    "expired_agent_command_sweep": "reconciler",
    "event_snapshots": "projector",
    "events_archive": "projector",
    "privacy_purge": "api",
    "privacy_deletion_requests": "api",
    "session_cleanup": "api",
    "fintrac_check": "api",
    "job_log_cleanup": "reconciler",
    "telemetry_retention": "reconciler",
    "notification_cleanup": "api",
    "user_images_pending_sweeper": "api",
    "user_images_hard_delete_gc": "api",
    "registry_health_probe": "reconciler",
    "snapshot_queue_retry": "api",
    "agent_rollout_watchdog": "api",
    "reaper_tick": "api",
    "lightning_watcher": "billing",
    "lightning_reconcile": "billing",
    "artifact_catalog_janitor": "api",
    "artifact_inventory_reconcile": "api",
}

DDL_TASKS = frozenset({"audit_partition_maintenance", "telemetry_partition_maintenance"})
