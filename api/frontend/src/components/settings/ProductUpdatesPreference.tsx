"use client";

import { useEffect, useState } from "react";
import { toast } from "sonner";
import { fetchEmailConsents, grantEmailConsent, withdrawEmailConsent } from "@/lib/api";
import { useLocale } from "@/lib/locale";
import { SettingsToggleRow } from "./settings-layout";

const PURPOSE = "product_updates";

/**
 * Opt in to, or out of, the monthly product-update email.
 *
 * This is consent to a commercial message (CASL), so it is off until the
 * person turns it on, and it saves the moment it is toggled rather than
 * waiting for the page's Save button: an unsubscribe must not be lost to an
 * unsaved form.
 */
export function ProductUpdatesPreference() {
  const { t } = useLocale();
  const [enabled, setEnabled] = useState(false);
  const [ready, setReady] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let live = true;
    fetchEmailConsents()
      .then((res) => {
        if (!live) return;
        const record = (res.consents ?? []).find((c) => c.purpose === PURPOSE);
        const expired = !!record?.expires_at && Date.parse(record.expires_at) <= Date.now();
        setEnabled(!!record?.active && !expired);
        setReady(true);
      })
      // Left disabled: showing "off" for a preference we could not read
      // would invite someone to switch on what may already be on.
      .catch(() => {});
    return () => { live = false; };
  }, []);

  const toggle = async () => {
    const next = !enabled;
    setSaving(true);
    try {
      if (next) await grantEmailConsent(PURPOSE);
      else await withdrawEmailConsent(PURPOSE);
      setEnabled(next);
      toast.success(t(next ? "dash.settings.product_updates_on" : "dash.settings.product_updates_off"));
    } catch {
      toast.error(t("dash.settings.product_updates_failed"));
    } finally {
      setSaving(false);
    }
  };

  return (
    <SettingsToggleRow
      title={t("dash.settings.product_updates")}
      description={t("dash.settings.product_updates_desc")}
      enabled={enabled}
      onToggle={() => void toggle()}
      disabled={!ready || saving}
    />
  );
}
