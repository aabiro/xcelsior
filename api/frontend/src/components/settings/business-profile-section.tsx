"use client";

import { useId, useRef, useState } from "react";
import { Building2, Check, ChevronDown, ImageUp, Loader2, Save, Trash2, Upload } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input, Label } from "@/components/ui/input";
import { SettingsSection } from "@/components/settings/settings-layout";
import { BusinessLogo } from "@/components/user/business-logo";
import {
  BusinessMenuCard,
  BusinessTriggerIdentity,
  type LogoOverride,
} from "@/components/user/business-identity";
import { deleteBusinessLogo, updateProfile, uploadBusinessLogo } from "@/lib/api";
import type { User } from "@/lib/auth";
import {
  DISPLAY_NAME_MAX,
  LOGO_ACCEPT,
  LOGO_MAX_BYTES,
  TAGLINE_MAX,
  cleanLine,
  normalizeTile,
  websiteIsValid,
  type BusinessIdentity,
} from "@/lib/business-profile";
import { useLocale } from "@/lib/locale";
import { cn } from "@/lib/utils";

const LOGO_TYPES = new Set(LOGO_ACCEPT.split(","));

/** Letters, not UTF-16 code units, as the API counts them. */
const lengthOf = (value: string) => Array.from(value).length;

/** "…your legal name, North Labs." without doubling the stop of "Acme Inc." */
function legalNameHint(t: (key: string, vars?: Record<string, string>) => string, legalName: string) {
  const name = legalName.trim();
  return t("dash.business.display_name_hint", { name }) + (/[.!?]$/.test(name) ? "" : ".");
}

/** Resolve once the browser has the image, so swapping to it does not flash.
 *  Never waits more than a few seconds: a slow image is better than a spinner
 *  that never stops. */
function preload(url: string, timeoutMs = 4000): Promise<void> {
  return new Promise((resolve) => {
    const timer = window.setTimeout(resolve, timeoutMs);
    const done = () => {
      window.clearTimeout(timer);
      resolve();
    };
    const image = new Image();
    image.onload = done;
    image.onerror = done;
    image.src = url;
  });
}

function Field({
  id,
  label,
  optional,
  counter,
  hint,
  error,
  children,
}: {
  id: string;
  label: string;
  optional?: string;
  counter?: { value: number; max: number };
  hint?: string;
  error?: string;
  children: React.ReactNode;
}) {
  const near = counter && counter.value >= counter.max * 0.85;
  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline justify-between gap-3">
        <Label htmlFor={id}>
          {label}
          {/* A real space, so a screen reader says "Website, optional", not "Websiteoptional". */}
          {optional && <> <span className="ml-0.5 text-xs font-normal text-text-muted">{optional}</span></>}
        </Label>
        {counter && (
          <span
            aria-hidden
            className={cn(
              "text-[11px] tabular-nums transition-colors",
              counter.value > counter.max ? "text-accent-red" : near ? "text-accent-gold" : "text-text-muted",
            )}
          >
            {counter.value}/{counter.max}
          </span>
        )}
      </div>
      {children}
      {error ? (
        <p id={`${id}-error`} className="text-xs text-accent-red">{error}</p>
      ) : hint ? (
        <p id={`${id}-hint`} className="text-xs leading-relaxed text-text-muted">{hint}</p>
      ) : null}
    </div>
  );
}

function PreviewFrame({
  theme,
  label,
  business,
  user,
  logo,
}: {
  theme: "light" | "dark";
  label: string;
  business: Partial<BusinessIdentity>;
  user: User;
  logo?: LogoOverride;
}) {
  const { t } = useLocale();
  return (
    <div
      data-theme={theme}
      className="dashboard-theme-scope business-preview-frame relative isolate overflow-hidden rounded-xl border border-[var(--edge)] p-3"
    >
      <span className="absolute left-3 top-2.5 text-[10px] font-semibold uppercase tracking-[0.1em] text-[var(--text-5)]">
        {label}
      </span>
      {/* The app bar, right-aligned as in the dashboard. */}
      <div className="mt-5 flex justify-end">
        <div className="flex min-w-0 items-center gap-2 rounded-full border border-[var(--line)] bg-[var(--panel)] py-1.5 pl-2 pr-2.5">
          <BusinessTriggerIdentity
            business={business}
            person={user}
            badgeLabel={t("dash.business.badge")}
            logo={logo}
            textClassName="block"
            compactName={false}
          />
          <ChevronDown className="h-4 w-4 shrink-0 text-[var(--text-4)]" aria-hidden />
        </div>
      </div>
      {/* The top of the account menu. */}
      <div className="dashboard-site-popout relative mt-2.5 ml-auto w-full max-w-[18rem] rounded-[18px] px-3.5 py-3">
        <BusinessMenuCard
          business={business}
          badgeLabel={t("dash.business.badge")}
          visitLabel={(site) => t("dash.business.visit_site", { site })}
          logo={logo}
          linkWebsite={false}
        />
      </div>
    </div>
  );
}

/** A business account's display name, website, tagline and logo. */
export function BusinessProfileSection({
  user,
  onUpdated,
}: {
  user: User;
  onUpdated: () => Promise<void>;
}) {
  const { t } = useLocale();
  const id = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [displayName, setDisplayName] = useState(user.business_display_name || "");
  const [website, setWebsite] = useState(user.business_website || "");
  const [tagline, setTagline] = useState(user.business_tagline || "");
  const [websiteEdited, setWebsiteEdited] = useState(false);
  const [websiteTouched, setWebsiteTouched] = useState(false);
  const [saving, setSaving] = useState(false);
  const [justSaved, setJustSaved] = useState(false);
  const [error, setError] = useState("");
  const [uploadPreview, setUploadPreview] = useState<string | null>(null);
  const [logoBusy, setLogoBusy] = useState<"upload" | "remove" | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const draft = {
    name: cleanLine(displayName),
    website: website.trim(),
    tagline: cleanLine(tagline),
  };
  const dirty =
    draft.name !== (user.business_display_name || "") ||
    draft.website !== (user.business_website || "") ||
    draft.tagline !== (user.business_tagline || "");
  const nameLength = lengthOf(draft.name);
  const taglineLength = lengthOf(draft.tagline);
  const websiteValid = websiteIsValid(draft.website);
  const valid = websiteValid && nameLength <= DISPLAY_NAME_MAX && taglineLength <= TAGLINE_MAX;
  const websiteError = websiteEdited && websiteTouched && !websiteValid ? t("dash.business.website_invalid") : undefined;

  const hasLogo = Boolean(user.business_logo_url);
  const tile = normalizeTile(user.business_logo_tile);
  const logoOverride: LogoOverride | undefined = uploadPreview ? { src: uploadPreview, tile: "none" } : undefined;

  // What the preview shows: the unsaved text, the saved (or uploading) logo.
  const previewBusiness: Partial<BusinessIdentity> = {
    account_type: "business",
    business_name: user.business_name,
    business_display_name: draft.name,
    business_tagline: draft.tagline,
    business_website:
      draft.website && websiteValid
        ? draft.website.includes("://") ? draft.website : `https://${draft.website}`
        : websiteValid ? "" : user.business_website,
    business_logo_url: user.business_logo_url,
    business_logo_tile: user.business_logo_tile,
  };

  const save = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!valid) {
      setWebsiteEdited(true);
      setWebsiteTouched(true);
      return;
    }
    if (!dirty || saving) return;
    setSaving(true);
    setError("");
    setJustSaved(false);
    try {
      const result = await updateProfile({
        business_display_name: draft.name,
        business_website: draft.website,
        business_tagline: draft.tagline,
      });
      // Show what was stored: the API adds https:// and tidies spacing.
      setDisplayName(result.business_display_name ?? draft.name);
      setWebsite(result.business_website ?? draft.website);
      setTagline(result.business_tagline ?? draft.tagline);
      setWebsiteEdited(false);
      setWebsiteTouched(false);
      await onUpdated();
      setJustSaved(true);
    } catch (err) {
      setError(err instanceof Error && err.message ? err.message : t("dash.business.save_failed"));
    } finally {
      setSaving(false);
    }
  };

  const uploadLogo = async (file: File | null | undefined) => {
    if (!file || logoBusy) return;
    if (!LOGO_TYPES.has(file.type)) {
      toast.error(t("dash.business.logo_type"));
      return;
    }
    if (file.size > LOGO_MAX_BYTES) {
      toast.error(t("dash.business.logo_too_big"));
      return;
    }
    const objectUrl = URL.createObjectURL(file);
    setUploadPreview(objectUrl);
    setLogoBusy("upload");
    try {
      const result = await uploadBusinessLogo(file);
      await preload(result.business_logo_url);
      await onUpdated();
      toast.success(t("dash.business.logo_uploaded"));
    } catch (err) {
      toast.error(err instanceof Error && err.message ? err.message : t("dash.business.logo_failed"));
    } finally {
      setUploadPreview(null);
      URL.revokeObjectURL(objectUrl);
      setLogoBusy(null);
    }
  };

  const removeLogo = async () => {
    if (logoBusy) return;
    setLogoBusy("remove");
    try {
      await deleteBusinessLogo();
      await onUpdated();
      toast.success(t("dash.business.logo_removed"));
    } catch (err) {
      toast.error(err instanceof Error && err.message ? err.message : t("dash.business.logo_remove_failed"));
    } finally {
      setLogoBusy(null);
    }
  };

  return (
    <SettingsSection
      id="business"
      icon={Building2}
      title={t("dash.business.profile")}
      description={t("dash.business.profile_desc")}
      accent="violet"
    >
      <form noValidate onSubmit={save} className="space-y-6">
        {/* ── Logo ──
            Phone: logo beside its hints, the drop zone full width below.
            Wider: logo on the left, hints and drop zone stacked beside it. */}
        <div className="grid grid-cols-[auto_minmax(0,1fr)] items-start gap-x-5 gap-y-4">
          <div className="group relative sm:row-span-2">
            <BusinessLogo
              business={previewBusiness}
              size="xl"
              src={logoOverride?.src}
              tile={logoOverride?.tile}
              className={cn("transition-opacity", logoBusy && "opacity-60")}
            />
            <button
              type="button"
              disabled={Boolean(logoBusy)}
              onClick={() => inputRef.current?.click()}
              aria-label={hasLogo ? t("dash.business.logo_change") : t("dash.business.logo_upload")}
              className={cn(
                "absolute inset-0 flex items-center justify-center rounded-[24px] bg-[rgba(5,7,14,0.55)]",
                "opacity-0 transition-opacity group-hover:opacity-100",
                "focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-cyan",
                logoBusy && "opacity-100 bg-[rgba(5,7,14,0.35)]",
              )}
            >
              {logoBusy ? (
                <Loader2 className="h-7 w-7 animate-spin text-white" />
              ) : (
                <ImageUp className="h-7 w-7 text-white drop-shadow" />
              )}
            </button>
            <input
              ref={inputRef}
              type="file"
              accept={LOGO_ACCEPT}
              className="sr-only"
              tabIndex={-1}
              aria-hidden
              onChange={(event) => {
                void uploadLogo(event.target.files?.[0]);
                event.target.value = "";
              }}
            />
          </div>

          <div className="pt-1">
            <p className="text-sm font-medium text-text-primary">{t("dash.business.logo")}</p>
            <ul className="mt-1 space-y-0.5 text-xs text-text-muted">
              <li>{t("dash.business.logo_hint_size")}</li>
              <li>{t("dash.business.logo_hint_transparent")}</li>
              <li>{t("dash.business.logo_hint_format")}</li>
            </ul>
          </div>

          <div className="col-span-2 space-y-3 sm:col-span-1 sm:col-start-2">
            <div
              onDragOver={(event) => {
                event.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(event) => {
                event.preventDefault();
                setDragOver(false);
                void uploadLogo(event.dataTransfer.files?.[0]);
              }}
              className={cn(
                "rounded-xl border border-dashed px-4 py-4 text-center transition-colors",
                dragOver
                  ? "border-accent-violet bg-accent-violet/8"
                  : "border-border/70 bg-surface/40 hover:border-border-light",
              )}
            >
              <Upload className="mx-auto mb-1.5 h-5 w-5 text-text-muted" aria-hidden />
              <p className="text-sm text-text-secondary">
                {t("dash.business.logo_drop")}{" "}
                <button
                  type="button"
                  className="font-medium text-accent-cyan hover:underline"
                  onClick={() => inputRef.current?.click()}
                  disabled={Boolean(logoBusy)}
                >
                  {t("dash.business.logo_browse")}
                </button>
              </p>
            </div>

            {hasLogo && !uploadPreview && (
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <p className="flex items-start gap-2 text-xs leading-relaxed text-text-muted">
                  <span
                    aria-hidden
                    data-tile={tile}
                    className="business-logo mt-[3px] h-3 w-3 shrink-0 rounded-[4px]"
                  />
                  {t(`dash.business.tile_${tile}`)}
                </p>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  disabled={Boolean(logoBusy)}
                  onClick={() => void removeLogo()}
                  className="shrink-0 self-start border-border text-text-muted hover:border-accent-red/40 hover:text-accent-red sm:self-auto"
                >
                  {logoBusy === "remove" ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <Trash2 className="h-3.5 w-3.5" />
                  )}
                  {t("dash.business.logo_remove")}
                </Button>
              </div>
            )}
          </div>
        </div>

        <div className="h-px bg-border/50" />

        {/* ── Name, website, tagline ── */}
        <div className="grid gap-4 sm:grid-cols-2">
          <Field
            id={`${id}-name`}
            label={t("dash.business.display_name")}
            optional={t("dash.business.optional")}
            counter={nameLength > DISPLAY_NAME_MAX * 0.6 ? { value: nameLength, max: DISPLAY_NAME_MAX } : undefined}
            hint={user.business_name ? legalNameHint(t, user.business_name) : undefined}
          >
            <Input
              id={`${id}-name`}
              value={displayName}
              maxLength={DISPLAY_NAME_MAX}
              placeholder={user.business_name || ""}
              autoComplete="organization"
              aria-describedby={user.business_name ? `${id}-name-hint` : undefined}
              onChange={(event) => {
                setDisplayName(event.target.value);
                setJustSaved(false);
              }}
            />
          </Field>
          <Field
            id={`${id}-website`}
            label={t("dash.business.website")}
            optional={t("dash.business.optional")}
            error={websiteError}
          >
            <Input
              id={`${id}-website`}
              value={website}
              inputMode="url"
              autoComplete="url"
              autoCapitalize="none"
              spellCheck={false}
              placeholder="example.com"
              aria-invalid={websiteError ? "true" : "false"}
              aria-describedby={websiteError ? `${id}-website-error` : undefined}
              className={cn(websiteError && "border-accent-red/60 focus:ring-accent-red/40")}
              onBlur={() => { if (websiteEdited) setWebsiteTouched(true); }}
              onChange={(event) => {
                setWebsiteEdited(true);
                setWebsite(event.target.value);
                setJustSaved(false);
              }}
            />
          </Field>
          <div className="sm:col-span-2">
            <Field
              id={`${id}-tagline`}
              label={t("dash.business.tagline")}
              optional={t("dash.business.optional")}
              counter={{ value: taglineLength, max: TAGLINE_MAX }}
            >
              <Input
                id={`${id}-tagline`}
                value={tagline}
                maxLength={TAGLINE_MAX}
                placeholder={t("dash.business.tagline_placeholder")}
                onChange={(event) => {
                  setTagline(event.target.value);
                  setJustSaved(false);
                }}
              />
            </Field>
          </div>
        </div>

        {/* ── Preview ── */}
        <div className="@container space-y-2.5">
          <div>
            <p className="text-sm font-medium text-text-primary">{t("dash.business.preview")}</p>
            <p className="text-xs text-text-muted">{t("dash.business.preview_hint")}</p>
          </div>
          <div className="grid gap-3 @[44rem]:grid-cols-2">
            <PreviewFrame
              theme="light"
              label={t("dash.business.preview_light")}
              business={previewBusiness}
              user={user}
              logo={logoOverride}
            />
            <PreviewFrame
              theme="dark"
              label={t("dash.business.preview_dark")}
              business={previewBusiness}
              user={user}
              logo={logoOverride}
            />
          </div>
        </div>

        {error && (
          <p role="alert" className="rounded-lg border border-accent-red/25 bg-accent-red/5 p-3 text-sm text-accent-red">
            {error}
          </p>
        )}
        <div className="flex flex-wrap items-center justify-end gap-3">
          {justSaved && !dirty && (
            <p role="status" className="inline-flex items-center gap-1.5 text-sm text-emerald">
              <Check className="h-4 w-4" aria-hidden />
              {t("dash.business.saved")}
            </p>
          )}
          <Button type="submit" disabled={saving || !dirty || !valid} className="w-full sm:w-auto">
            {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
            {t("dash.business.save")}
          </Button>
        </div>
      </form>
    </SettingsSection>
  );
}

/** What a personal account sees: what a business profile is, and the way in. */
export function BusinessProfileTeaser() {
  const { t } = useLocale();
  const focusAccountType = () => {
    const select = document.querySelector<HTMLSelectElement>('[data-field="account-type"]');
    if (!select) return;
    select.scrollIntoView({ behavior: "smooth", block: "center" });
    select.focus({ preventScroll: true });
  };
  return (
    <section
      id="business"
      className="scroll-mt-4 flex flex-col gap-4 rounded-2xl border border-dashed border-border/70 bg-surface/50 p-5 sm:flex-row sm:items-center"
    >
      <div className="flex min-w-0 flex-1 items-start gap-3">
        <BusinessLogo size="md" />
        <div className="min-w-0">
          <h3 className="text-sm font-semibold text-text-primary">{t("dash.business.profile")}</h3>
          <p className="mt-0.5 text-xs leading-relaxed text-text-muted">{t("dash.business.locked_desc")}</p>
        </div>
      </div>
      <Button type="button" variant="outline" size="sm" onClick={focusAccountType} className="shrink-0 self-start sm:self-center">
        {t("dash.business.locked_action")}
      </Button>
    </section>
  );
}
