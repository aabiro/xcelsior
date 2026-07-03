const BRAND_SYSTEM_BASE = "/brand-system";
const SITE_ASSET_BASE = "/site-assets/assets";
const SITE_ASSET_PNG_BASE = `${SITE_ASSET_BASE}/png`;
const SITE_ICON_BASE = `${SITE_ASSET_BASE}/icons`;

export const SITE_ASSETS = {
  appCircleDark: `${SITE_ASSET_BASE}/app-circle-dark.svg`,
  appCircleLight: `${SITE_ASSET_BASE}/app-circle-light.svg`,
  appGradientCircle: `${SITE_ASSET_BASE}/app-gradient-circle.svg`,
  appGradientRounded: `${SITE_ASSET_BASE}/app-gradient-rounded.svg`,
  appRoundedDark: `${SITE_ASSET_BASE}/app-rounded-dark.svg`,
  appRoundedLight: `${SITE_ASSET_BASE}/app-rounded-light.svg`,
  appSquareDark: `${SITE_ASSET_BASE}/app-square-dark.svg`,
  appSquareLight: `${SITE_ASSET_BASE}/app-square-light.svg`,
  iconGradient: `${SITE_ASSET_BASE}/icon-gradient.svg`,
  iconGradientTight: `${BRAND_SYSTEM_BASE}/icon-gradient-tight.svg`,
  iconNavy: `${SITE_ASSET_BASE}/icon-navy.svg`,
  iconWhite: `${SITE_ASSET_BASE}/icon-white.svg`,
  lockupDark: `${SITE_ASSET_BASE}/lockup-dark.svg`,
  lockupLight: `${SITE_ASSET_BASE}/lockup-light.svg`,
  wordmarkDark: `${SITE_ASSET_BASE}/wordmark-dark.svg`,
  wordmarkLight: `${SITE_ASSET_BASE}/wordmark-light.svg`,
  appGradientRounded512: `${SITE_ASSET_PNG_BASE}/app-gradient-rounded-512.png`,
  appleTouchIcon180: `${SITE_ASSET_PNG_BASE}/apple-touch-icon-180.png`,
  facebookCover1640x624: `${SITE_ASSET_PNG_BASE}/facebook-cover-1640x624.png`,
  favicon16: `${SITE_ASSET_PNG_BASE}/favicon-16.png`,
  favicon32: `${SITE_ASSET_PNG_BASE}/favicon-32.png`,
  favicon48: `${SITE_ASSET_PNG_BASE}/favicon-48.png`,
  icon192: `${SITE_ASSET_PNG_BASE}/icon-192.png`,
  icon512: `${SITE_ASSET_PNG_BASE}/icon-512.png`,
  iconGradient512: `${SITE_ASSET_PNG_BASE}/icon-gradient-512.png`,
  iconMaskable512: `${SITE_ASSET_PNG_BASE}/icon-maskable-512.png`,
  iconNavy512: `${SITE_ASSET_PNG_BASE}/icon-navy-512.png`,
  iconWhite512: `${SITE_ASSET_PNG_BASE}/icon-white-512.png`,
  instagramPortrait1080x1350: `${SITE_ASSET_PNG_BASE}/instagram-portrait-1080x1350.png`,
  instagramPost1080: `${SITE_ASSET_PNG_BASE}/instagram-post-1080.png`,
  linkedinBanner1584x396: `${SITE_ASSET_PNG_BASE}/linkedin-banner-1584x396.png`,
  ogImage1200x630: `${SITE_ASSET_PNG_BASE}/og-image-1200x630.png`,
  profileDark1000: `${SITE_ASSET_PNG_BASE}/profile-dark-1000.png`,
  profileGradient1000: `${SITE_ASSET_PNG_BASE}/profile-gradient-1000.png`,
  twitterHeader1500x500: `${SITE_ASSET_PNG_BASE}/twitter-header-1500x500.png`,
  xPost1600x900: `${SITE_ASSET_PNG_BASE}/x-post-1600x900.png`,
} as const;

export const siteIcon = (name: string, theme: "dark" | "light") =>
  `${SITE_ICON_BASE}/${name}.${theme}.svg`;

export const BRAND_ASSETS = {
  appRoundedDark: SITE_ASSETS.appRoundedDark,
  appRoundedLight: SITE_ASSETS.appRoundedLight,
  iconGradient: SITE_ASSETS.iconGradient,
  iconGradientTight: `${BRAND_SYSTEM_BASE}/icon-gradient-tight.svg`,
  lockupDark: SITE_ASSETS.lockupDark,
  lockupLight: SITE_ASSETS.lockupLight,
  textEverMedDark: `${BRAND_SYSTEM_BASE}/text-ever-med-dark.svg`,
  textEverMedLight: `${BRAND_SYSTEM_BASE}/text-ever-med-light.svg`,
  textEverSbDark: `${BRAND_SYSTEM_BASE}/text-ever-sb-dark.svg`,
  textEverSbLight: `${BRAND_SYSTEM_BASE}/text-ever-sb-light.svg`,
  textTagMedDark: `${BRAND_SYSTEM_BASE}/text-tag-med-dark.svg`,
  textTagMedLight: `${BRAND_SYSTEM_BASE}/text-tag-med-light.svg`,
  textUrlMedDark: `${BRAND_SYSTEM_BASE}/text-url-med-dark.svg`,
  textUrlMedLight: `${BRAND_SYSTEM_BASE}/text-url-med-light.svg`,
} as const;

export const BRAND_PNG_ASSETS = {
  appGradientRounded512: SITE_ASSETS.appGradientRounded512,
  appleTouchIcon180: SITE_ASSETS.appleTouchIcon180,
  facebookCover1640x624: SITE_ASSETS.facebookCover1640x624,
  favicon16: SITE_ASSETS.favicon16,
  favicon32: SITE_ASSETS.favicon32,
  favicon48: SITE_ASSETS.favicon48,
  icon192: SITE_ASSETS.icon192,
  icon512: SITE_ASSETS.icon512,
  iconGradient512: SITE_ASSETS.iconGradient512,
  iconMaskable512: SITE_ASSETS.iconMaskable512,
  iconNavy512: SITE_ASSETS.iconNavy512,
  iconWhite512: SITE_ASSETS.iconWhite512,
  instagramPortrait1080x1350: SITE_ASSETS.instagramPortrait1080x1350,
  instagramPost1080: SITE_ASSETS.instagramPost1080,
  linkedinBanner1584x396: SITE_ASSETS.linkedinBanner1584x396,
  ogImage1200x630: SITE_ASSETS.ogImage1200x630,
  profileDark1000: SITE_ASSETS.profileDark1000,
  profileGradient1000: SITE_ASSETS.profileGradient1000,
  twitterHeader1500x500: SITE_ASSETS.twitterHeader1500x500,
  xPost1600x900: SITE_ASSETS.xPost1600x900,
} as const;

export const BRAND_ASSET_ORIGIN = "https://xcelsior.ca";
