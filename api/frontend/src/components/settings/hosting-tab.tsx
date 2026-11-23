"use client";

import Link from "next/link";
import { ChevronRight, Server, Wallet } from "lucide-react";
import type { User } from "@/lib/auth";
import { ProviderProfileForm } from "@/components/providers/seller-profile-card";
import { StartHostingButton } from "@/components/hosting/start-hosting-button";
import { SettingsSection, SettingsTabPanel } from "@/components/settings/settings-layout";
import { StaggerItem, StaggerList } from "@/components/ui/motion";
import { hostingEnabled } from "@/lib/hosting";
import { useLocale } from "@/lib/locale";

/**
 * GPU hosting's own Settings tab: turn hosting on, or go to the machines,
 * and the payout details hosting needs.
 *
 * The owner asked for account and hosting to be separate tabs (2026-11-23),
 * after one "Account & hosting" tab held both.
 */
export function HostingTab({ user, onUpdated }: { user: User; onUpdated: () => Promise<void> }) {
  const { t } = useLocale();
  const hosting = hostingEnabled(user);
  return (
    <SettingsTabPanel>
      <StaggerList className="space-y-5">
        <StaggerItem>
          <SettingsSection
            id="hosting"
            icon={Server}
            title={t("dash.settings.hosting_title")}
            description={t("dash.settings.hosting_desc")}
            accent="emerald"
            highlight
          >
            {hosting ? (
              <Link
                href="/dashboard/hosting"
                className="inline-flex min-h-11 items-center gap-1.5 text-sm font-medium text-accent-cyan hover:underline"
              >
                {t("dash.settings.hosting_manage")}
                <ChevronRight className="h-4 w-4" aria-hidden />
              </Link>
            ) : (
              <StartHostingButton />
            )}
          </SettingsSection>
        </StaggerItem>

        {hosting && (
          <StaggerItem>
            <SettingsSection
              id="payout"
              icon={Wallet}
              title={t("dash.hosting.payout_title")}
              description={t("dash.hosting.payout_desc")}
              accent="emerald"
            >
              <ProviderProfileForm showHeading={false} onSaved={() => { void onUpdated(); }} />
            </SettingsSection>
          </StaggerItem>
        )}
      </StaggerList>
    </SettingsTabPanel>
  );
}
