export const ONBOARDING_STEPS = [
  { key: "profile", labelKey: "gear.step_profile", descKey: "gear.step_profile_desc", href: "/dashboard/settings" },
  { key: "jurisdiction", labelKey: "gear.step_jurisdiction", descKey: "gear.step_jurisdiction_desc", href: "/dashboard/settings" },
  { key: "api_key", labelKey: "gear.step_api_key", descKey: "gear.step_api_key_desc", href: "/dashboard/settings#api-keys" },
  { key: "browse", labelKey: "gear.step_browse", descKey: "gear.step_browse_desc", href: "/dashboard/marketplace" },
  { key: "instance", labelKey: "gear.step_instance", descKey: "gear.step_instance_desc", href: "/dashboard/instances/new" },
] as const;

export type OnboardingStepKey = (typeof ONBOARDING_STEPS)[number]["key"];

export const AUTO_DETECTED_KEYS = new Set<OnboardingStepKey>([
  "profile",
  "api_key",
  "browse",
  "instance",
  "jurisdiction",
]);