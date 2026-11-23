"use client";

import Link from "next/link";
import { ChevronRight, IdCard, Server } from "lucide-react";
import type { User } from "@/lib/auth";
import { AccountProfileForm } from "@/components/ui/province-selector";
import { BusinessProfileSection } from "@/components/settings/business-profile-section";
import { ProviderProfileForm } from "@/components/providers/seller-profile-card";
import { StartHostingButton } from "@/components/hosting/start-hosting-button";
import { SettingsSection, SettingsTabPanel } from "@/components/settings/settings-layout";
import { StaggerItem, StaggerList } from "@/components/ui/motion";
import { isBusinessAccount } from "@/lib/business-profile";
import { hostingEnabled } from "@/lib/hosting";
import { useLocale } from "@/lib/locale";

/**
 * The account's home: the one Personal/Business switch, the business profile
 * that switch unlocks, and GPU hosting with its payout details.
 *
 * These used to sit inside the Profile tab under the photo, name and email,
 * with a "Change account type" button and a hosting card that sent you back up
 * to the same select (owner review, 2026-11-22). Profile is now only profile;
 * this tab is the single place to change account type or manage hosting.
 */
export function AccountHostingTab({ user, onUpdated }: { user: User; onUpdated: () => Promise<void> }) {
  const { t } = useLocale();
  const hosting = hostingEnabled(user);
  return (
    <SettingsTabPanel>
      <StaggerList className="space-y-5">
        <StaggerItem>
          <SettingsSection
            id="account"
            icon={IdCard}
            title={t("dash.settings.account_title")}
            description={t("dash.settings.account_desc")}
            accent="cyan"
            highlight
          >
            <AccountProfileForm settings />
          </SettingsSection>
        </StaggerItem>

        {isBusinessAccount(user) && (
          <StaggerItem>
            <BusinessProfileSection key={user.user_id} user={user} onUpdated={onUpdated} />
          </StaggerItem>
        )}

        <StaggerItem>
          <SettingsSection
            id="hosting"
            icon={Server}
            title={t("dash.settings.hosting_title")}
            description={t("dash.settings.hosting_desc")}
            accent="emerald"
          >
            <div className="space-y-6">
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
              {hosting && <ProviderProfileForm onSaved={() => { void onUpdated(); }} />}
            </div>
          </SettingsSection>
        </StaggerItem>
      </StaggerList>
    </SettingsTabPanel>
  );
}
