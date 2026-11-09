"use client";

import { useEffect, useId, useState } from "react";
import Link from "next/link";
import { ArrowRight, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input, Label } from "@/components/ui/input";
import * as api from "@/lib/api";
import { useAuth, type User } from "@/lib/auth";
import { useLocale } from "@/lib/locale";
import { cn } from "@/lib/utils";

type SellerFields = {
  legal_name?: string;
  address_line1?: string;
  address_line2?: string;
  city?: string;
  province?: string;
  postal_code?: string;
  country?: string;
  tax_id_kind?: string;
  tax_id_last4?: string;
  gst_hst_number?: string;
  seller_profile_complete?: boolean;
};

type Address = {
  address_line1: string;
  address_line2: string;
  city: string;
  province: string;
  postal_code: string;
  country: string;
};

const TAX_KINDS = [
  ["sin", "dash.earnings.seller_tax_sin"],
  ["bn", "dash.earnings.seller_tax_bn"],
  ["gst", "dash.earnings.seller_tax_gst"],
  ["other", "dash.earnings.seller_tax_other"],
] as const;

function accountAddress(user: User | null | undefined, provider: SellerFields | null): Address {
  const source = user?.address_line1 ? user : provider;
  return {
    address_line1: source?.address_line1 || "",
    address_line2: source?.address_line2 || "",
    city: source?.city || "",
    province: source?.province || "",
    postal_code: source?.postal_code || "",
    country: source?.country || "",
  };
}

function addressReady(address: Address) {
  return Boolean(
    address.address_line1.trim() &&
      address.city.trim() &&
      address.postal_code.trim() &&
      address.country.trim(),
  );
}

function addressLines(address: Address) {
  const street = [address.address_line1, address.address_line2].map((part) => part.trim()).filter(Boolean).join(", ");
  const place = [address.city, address.province, address.postal_code].map((part) => part.trim()).filter(Boolean).join(" ");
  return [street, place, address.country.trim()].filter(Boolean);
}

export function ProviderProfileForm({
  onSaved,
  mode = "settings",
  onEditAddress,
}: {
  onSaved: () => void | Promise<void>;
  mode?: "onboarding" | "settings";
  onEditAddress?: () => void;
}) {
  const [profile, setProfile] = useState<SellerFields | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    api.apiFetch<{profile: SellerFields}>("/api/providers/seller-profile")
      .then((result) => {
        if (!active) return;
        const saved = result.profile || {};
        setProfile({
          ...saved,
          legal_name: saved.legal_name || "",
        });
      }).catch((err) => { if (active) setError(err instanceof Error ? err.message : "Unable to load provider profile"); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, []);
  if (loading) return <p role="status" className="text-sm text-text-secondary">Loading provider details…</p>;
  if (error) return <p role="alert" className="text-sm text-red-400">{error}</p>;
  return <SellerProfileCard provider={profile} onSaved={onSaved} mode={mode} onEditAddress={onEditAddress} />;
}

export function SellerProfileCard({
  provider,
  onSaved,
  mode = "settings",
  onEditAddress,
}: {
  provider: SellerFields | null;
  onSaved: () => void | Promise<void>;
  mode?: "onboarding" | "settings";
  onEditAddress?: () => void;
}) {
  const { t } = useLocale();
  const { user } = useAuth();
  const [legalName, setLegalName] = useState(provider?.legal_name || "");
  const [kind, setKind] = useState(provider?.tax_id_kind || "");
  const [taxId, setTaxId] = useState("");
  const [gst, setGst] = useState(provider?.gst_hst_number || "");
  const taxGroup = useId();
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const onboarding = mode === "onboarding";
  const address = accountAddress(user, provider);
  const readyAddress = addressReady(address);

  const save = async () => {
    const name = legalName.trim();
    if (name.length < 2) {
      setError(t("dash.earnings.seller_name_error"));
      return;
    }
    if (!kind) {
      setError(t("dash.earnings.seller_kind_error"));
      return;
    }
    if (!taxId.trim() && !provider?.tax_id_last4) {
      setError(t("dash.earnings.seller_id_error"));
      return;
    }
    if (!readyAddress) {
      setError(t("dash.earnings.seller_address_missing"));
      return;
    }
    setSaving(true);
    setError("");
    try {
      const gstNumber = kind === "gst" ? taxId.trim() : kind === "sin" || kind === "bn" ? gst.trim() : "";
      await api.saveSellerProfile({
        legal_name: name,
        ...address,
        tax_id_kind: kind,
        tax_id: taxId,
        gst_hst_number: gstNumber,
        accepted_terms: true,
      });
      setTaxId("");
      toast.success(t("dash.earnings.seller_saved"));
      await onSaved();
    } catch (err) {
      const message = err instanceof Error ? err.message : t("dash.earnings.seller_save");
      setError(message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form className={cn("space-y-7", !onboarding && "border-t border-border/50 pt-6")} noValidate onSubmit={(event) => { event.preventDefault(); void save(); }}>
      {!onboarding && (
        <div className="space-y-1">
          <h3 className="text-base font-semibold tracking-tight">{t("dash.earnings.seller_title")}</h3>
          <p className="text-sm text-text-secondary">{t("dash.earnings.seller_desc")}</p>
        </div>
      )}
      <Field label={t("dash.earnings.seller_legal_name")} value={legalName} onChange={setLegalName} autoComplete="off" placeholder="" />
      <fieldset className="space-y-3">
        <legend className="text-sm font-medium text-text-secondary">{t("dash.earnings.seller_tax_kind")}</legend>
        <div className="grid grid-cols-2 gap-2">
          {TAX_KINDS.map(([value, label]) => (
            <label key={value} className={cn(
              "flex cursor-pointer items-center justify-center rounded-2xl border px-3 py-4 text-center text-sm font-medium transition-colors",
              kind === value ? "border-accent-cyan/50 bg-accent-cyan/10 text-text-primary" : "border-border bg-navy/30 text-text-secondary hover:bg-surface-hover",
            )}>
              <input className="sr-only" type="radio" name={taxGroup} value={value} checked={kind === value} onChange={() => setKind(value)} />
              {t(label)}
            </label>
          ))}
        </div>
        <p className="text-xs leading-relaxed text-text-muted">{t("dash.earnings.seller_tax_accepted")}</p>
      </fieldset>
      <Field
        label={t("dash.earnings.seller_tax_id")}
        value={taxId}
        onChange={setTaxId}
        autoComplete="off"
        placeholder=""
        hint={[kind === "gst" ? t("dash.earnings.seller_gst_example") : "", provider?.tax_id_last4 ? `${t("dash.earnings.seller_on_file")} ${provider.tax_id_last4}` : ""].filter(Boolean).join(" ") || undefined}
        className="[&_input]:font-mono [&_input]:tracking-wide"
      />
      {(kind === "sin" || kind === "bn") && (
        <Field
          label={t("dash.earnings.seller_gst")}
          value={gst}
          onChange={setGst}
          autoComplete="off"
          placeholder=""
          optional={t("dash.earnings.seller_gst_hint")}
          hint={t("dash.earnings.seller_gst_example")}
        />
      )}
      {onboarding ? (
        <div className="flex items-start justify-between gap-4 rounded-2xl border border-border bg-navy/30 px-4 py-4">
          <div className="min-w-0">
            <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-accent-cyan">{t("dash.earnings.seller_address_on_file")}</p>
            {readyAddress ? (
              <p className="mt-2 text-sm leading-relaxed text-text-primary">{addressLines(address).join(" · ")}</p>
            ) : (
              <p className="mt-2 text-sm text-text-secondary">{t("dash.earnings.seller_address_missing")}</p>
            )}
          </div>
          {onEditAddress && (
            <button type="button" onClick={onEditAddress} className="shrink-0 rounded-full border border-border px-3 py-1 text-xs font-medium text-text-secondary transition-colors hover:text-text-primary">
              {t("dash.earnings.seller_address_edit")}
            </button>
          )}
        </div>
      ) : (
        <p className="text-xs text-text-muted">{t("dash.earnings.seller_address_uses_account")}</p>
      )}
      {error && <p role="alert" className="rounded-xl border border-red-400/25 bg-red-400/5 px-4 py-3 text-sm text-red-400">{error}</p>}
      <div className="space-y-3">
        <Button className={onboarding ? "h-12 w-full text-base" : ""} type="submit" disabled={saving}>
          {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
          {saving ? t("dash.earnings.seller_saving") : onboarding ? t("dash.earnings.seller_continue") : t("dash.earnings.seller_save")}
          {onboarding && !saving ? <ArrowRight className="h-4 w-4" /> : null}
        </Button>
        <p className="text-center text-xs text-text-muted">
          <Link href="/terms#provider-terms" className="underline decoration-border underline-offset-4 transition-colors hover:text-text-secondary">
            {t("dash.earnings.seller_terms")}
          </Link>
        </p>
      </div>
    </form>
  );
}

function Field({
  label,
  value,
  onChange,
  hint,
  optional,
  autoComplete,
  placeholder,
  className,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  hint?: string;
  optional?: string;
  autoComplete?: string;
  placeholder?: string;
  className?: string;
}) {
  const id = useId();
  return (
    <div className={cn("space-y-2", className)}>
      <div className="flex items-baseline justify-between gap-3">
        <Label htmlFor={id}>{label}</Label>
        {optional && <span className="text-xs text-text-muted">{optional}</span>}
      </div>
      <Input id={id} value={value} placeholder={placeholder} onChange={(event) => onChange(event.target.value)} autoComplete={autoComplete} className="h-11" />
      {hint && <p className="text-xs leading-relaxed text-text-muted">{hint}</p>}
    </div>
  );
}
