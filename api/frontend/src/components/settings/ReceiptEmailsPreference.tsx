"use client";

import { useEffect, useState } from "react";
import { toast } from "sonner";
import { apiFetch } from "@/lib/api";
import { useLocale } from "@/lib/locale";
import { SettingsToggleRow } from "./settings-layout";

/**
 * The receipt email sent when an instance finishes: billed seconds, the rate
 * locked at launch, and what came out of the wallet.
 *
 * It is a service message about the account's own spending, so it is on by
 * default (billing/receipt_emails.py). It saves the moment it is toggled, like
 * the other email switches, and it reads as off while email notifications
 * are off, because the sweep sends nothing then either.
 */
export function ReceiptEmailsPreference({ emailOn = true }: { emailOn?: boolean }) {
  const { t } = useLocale();
  const [enabled, setEnabled] = useState(true);
  const [ready, setReady] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let live = true;
    apiFetch<{ receipt_emails?: boolean }>("/api/users/me/preferences")
      .then((prefs) => {
        if (!live) return;
        setEnabled(prefs.receipt_emails !== false);
        setReady(true);
      })
      // Left disabled: a switch for a setting we could not read would
      // invite someone to change what they cannot see.
      .catch(() => {});
    return () => { live = false; };
  }, []);

  const toggle = async () => {
    const next = !enabled;
    setSaving(true);
    try {
      await apiFetch("/api/users/me/preferences", {
        method: "PUT",
        body: JSON.stringify({ receipt_emails: next }),
      });
      setEnabled(next);
      toast.success(t(next ? "dash.settings.receipt_emails_on" : "dash.settings.receipt_emails_off"));
    } catch {
      toast.error(t("dash.settings.receipt_emails_failed"));
    } finally {
      setSaving(false);
    }
  };

  return (
    <SettingsToggleRow
      title={t("dash.settings.receipt_emails")}
      description={t(emailOn ? "dash.settings.receipt_emails_desc" : "dash.settings.receipt_emails_needs_email")}
      enabled={emailOn && enabled}
      onToggle={() => void toggle()}
      disabled={!emailOn || !ready || saving}
    />
  );
}
