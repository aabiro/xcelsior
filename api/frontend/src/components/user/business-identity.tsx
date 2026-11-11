"use client";

import { ArrowUpRight } from "lucide-react";
import { cn } from "@/lib/utils";
import type { BusinessLogoTile, User } from "@/lib/auth";
import { BusinessBadge, BusinessLogo } from "@/components/user/business-logo";
import { UserAvatar } from "@/components/user/user-avatar";
import { businessName, websiteLabel, type BusinessIdentity } from "@/lib/business-profile";

/**
 * The business as the app bar and the account menu show it. The settings
 * preview renders these same components with the unsaved values, so what it
 * shows is what the dashboard will show.
 */

type Person = Pick<User, "name" | "email" | "avatar_url" | "is_admin">;

export interface LogoOverride {
  src: string | null;
  tile: BusinessLogoTile | null;
}

/** Logo with the person's avatar on its corner, then name, badge and person. */
export function BusinessTriggerIdentity({
  business,
  person,
  badgeLabel,
  logo,
  textClassName = "hidden sm:block",
  compactName = true,
}: {
  business: Partial<BusinessIdentity>;
  person: Person;
  badgeLabel: string;
  logo?: LogoOverride;
  /** The text is hidden on a phone in the app bar; the preview always shows it. */
  textClassName?: string;
  /** In the app bar the name narrows below 1280 px, where the bar is full. */
  compactName?: boolean;
}) {
  const personName = person.name || person.email;
  return (
    <>
      <span className="relative shrink-0">
        <BusinessLogo business={business} size="md" src={logo?.src} tile={logo?.tile} />
        <UserAvatar
          user={person}
          size="2xs"
          showRing={false}
          className="business-identity-person absolute -bottom-1 -right-1"
        />
      </span>
      <div className={cn("min-w-0 text-left", textClassName)}>
        <p
          className={cn(
            "truncate text-base font-medium leading-none text-[var(--text)]",
            compactName ? "max-w-[9rem] xl:max-w-[14rem]" : "max-w-[14rem]",
          )}
        >
          {businessName(business)}
        </p>
        <p className="mt-1 flex min-w-0 items-center gap-1.5 text-sm leading-5 text-[var(--text-4)]">
          <BusinessBadge label={badgeLabel} />
          <span className="min-w-0 max-w-[10rem] truncate">
            {person.is_admin ? `${personName} · Admin` : personName}
          </span>
        </p>
      </div>
    </>
  );
}

/** The business at the top of the account menu: logo, name, tagline, website. */
export function BusinessMenuCard({
  business,
  badgeLabel,
  visitLabel,
  logo,
  linkWebsite = true,
}: {
  business: Partial<BusinessIdentity>;
  badgeLabel: string;
  /** Accessible name for the website link; receives the site's label. */
  visitLabel: (site: string) => string;
  logo?: LogoOverride;
  /** The preview shows the address without making it a working link. */
  linkWebsite?: boolean;
}) {
  const site = websiteLabel(business.business_website);
  const siteClass =
    "mt-1.5 inline-flex max-w-full items-center gap-1 align-top text-xs font-medium text-[var(--cyan)]";
  return (
    <div className="flex items-start gap-3">
      <BusinessLogo business={business} size="lg" src={logo?.src} tile={logo?.tile} />
      <div className="min-w-0 flex-1">
        <BusinessBadge label={badgeLabel} />
        <p className="mt-1.5 line-clamp-2 break-words text-sm font-semibold leading-snug text-[var(--text)]">
          {businessName(business)}
        </p>
        {business.business_tagline && (
          <p className="mt-0.5 line-clamp-2 break-words text-xs leading-snug text-[var(--text-4)]">
            {business.business_tagline}
          </p>
        )}
        {site &&
          (linkWebsite && business.business_website ? (
            <a
              href={business.business_website}
              target="_blank"
              rel="noopener noreferrer"
              aria-label={visitLabel(site)}
              className={cn(siteClass, "hover:underline")}
            >
              <span className="truncate">{site}</span>
              <ArrowUpRight className="h-3 w-3 shrink-0" aria-hidden />
            </a>
          ) : (
            <span className={siteClass}>
              <span className="truncate">{site}</span>
              <ArrowUpRight className="h-3 w-3 shrink-0" aria-hidden />
            </span>
          ))}
      </div>
    </div>
  );
}
