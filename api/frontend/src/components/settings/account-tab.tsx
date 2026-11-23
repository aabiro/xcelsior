"use client";

import { IdCard } from "lucide-react";
import type { User } from "@/lib/auth";
import { AccountProfileForm } from "@/components/ui/province-selector";
import { BusinessProfileSection } from "@/components/settings/business-profile-section";
import { SettingsSection, SettingsTabPanel } from "@/components/settings/settings-layout";
import { StaggerItem, StaggerList } from "@/components/ui/motion";
import { isBusinessAccount } from "@/lib/business-profile";
import { useLocale } from "@/lib/locale";

/**
 * The account: the one Personal/Business switch with its address, and the
 * business profile that switch unlocks.
 *
 * GPU hosting and its payout details are not here. They live on
 * /dashboard/hosting, hosting's own page; the owner asked for account and
 * hosting to be separate (2026-11-23), after an "Account & hosting" tab.
 */
export function AccountTab({ user, onUpdated }: { user: User; onUpdated: () => Promise<void> }) {
  const { t } = useLocale();
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
      </StaggerList>
    </SettingsTabPanel>
  );
}
