"use client";

import { useEffect, useId, useState } from "react";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input, Label } from "@/components/ui/input";
import * as api from "@/lib/api";
import { AddressLookup, LocationFields } from "@/components/ui/province-selector";
import { useAuth } from "@/lib/auth";
import { useLocale } from "@/lib/locale";

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

export function ProviderProfileForm({ onSaved }: { onSaved: () => void | Promise<void> }) {
  const { user } = useAuth();
  const [profile, setProfile] = useState<SellerFields | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    api.apiFetch<{profile: SellerFields}>("/api/providers/seller-profile")
      .then((result) => {
        if (!active) return;
        const saved = result.profile;
        const address = saved.country || saved.address_line1 ? saved : user;
        setProfile({ ...saved,
          legal_name: saved.legal_name || user?.business_name || user?.name || "",
          address_line1: address?.address_line1 || "",
          address_line2: address?.address_line2 || "",
          city: address?.city || "", postal_code: address?.postal_code || "",
          country: address?.country || "", province: address?.province || "",
        });
      }).catch((err) => { if (active) setError(err instanceof Error ? err.message : "Unable to load provider profile"); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [user]);
  if (loading) return <p role="status">Loading provider details…</p>;
  if (error) return <p role="alert">{error}</p>;
  return <SellerProfileCard provider={profile} onSaved={onSaved} />;
}

export function SellerProfileCard({
  provider,
  onSaved,
}: {
  provider: SellerFields | null;
  onSaved: () => void | Promise<void>;
}) {
  const { t } = useLocale();
  const [legalName, setLegalName] = useState(provider?.legal_name || "");
  const [line1, setLine1] = useState(provider?.address_line1 || "");
  const [line2, setLine2] = useState(provider?.address_line2 || "");
  const [city, setCity] = useState(provider?.city || "");
  const [province, setProvince] = useState(provider?.province || "");
  const [postal, setPostal] = useState(provider?.postal_code || "");
  const [country, setCountry] = useState(provider?.country || "");
  const [kind, setKind] = useState(provider?.tax_id_kind || "");
  const [taxId, setTaxId] = useState("");
  const [gst, setGst] = useState(provider?.gst_hst_number || "");
  const [accepted, setAccepted] = useState(Boolean(provider?.seller_profile_complete));
  const [requiredFields, setRequiredFields] = useState<string[] | null>(null);
  const taxKindId = useId();
  const [editRevision, setEditRevision] = useState(0);
  const manual = (set: (value: string) => void) => (value: string) => { setEditRevision((revision) => revision + 1); set(value); };
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  const save = async () => {
    setSaving(true); setError("");
    try {
      await api.saveSellerProfile({
        legal_name: legalName,
        address_line1: line1,
        address_line2: line2,
        city,
        province,
        postal_code: postal,
        country,
        tax_id_kind: kind,
        tax_id: taxId,
        gst_hst_number: gst,
        accepted_terms: accepted,
      });
      setTaxId("");
      toast.success(t("dash.earnings.seller_saved"));
      await onSaved();
    } catch (err) {
      const message=err instanceof Error ? err.message : t("dash.earnings.seller_save");
      setError(message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">{t("dash.earnings.seller_title")}</CardTitle>
        <CardDescription>{t("dash.earnings.seller_desc")}</CardDescription>
      </CardHeader>
      <CardContent><form className="space-y-3" onSubmit={(event) => { event.preventDefault(); void save(); }}>
        {provider?.tax_id_last4 && (
          <p className="text-xs text-text-muted">
            {t("dash.earnings.seller_on_file")} {provider.tax_id_last4}
          </p>
        )}
        <div className="grid gap-3 sm:grid-cols-2">
          <Field label={t("dash.earnings.seller_legal_name")} value={legalName} onChange={setLegalName} />
          <div className="sm:col-span-2"><LocationFields country={country} province={province}
            onCountryChange={setCountry} onProvinceChange={manual(setProvince)} onRequirementsChange={setRequiredFields} /></div>
          <div className="sm:col-span-2"><AddressLookup country={country} editRevision={editRevision} onSelect={(address) => {
            setLine1(address.address_line1); setLine2(address.address_line2); setCity(address.city);
            setPostal(address.postal_code); setCountry(address.country); setProvince(address.province);
          }} /></div>
          <Field label={t("dash.earnings.seller_address")} value={line1} onChange={manual(setLine1)} required={requiredFields?.includes("address_line1") ?? true} className="sm:col-span-2" />
          <Field label={t("dash.earnings.seller_address2")} value={line2} onChange={manual(setLine2)} required={false} className="sm:col-span-2" />
          <Field label={t("dash.earnings.seller_city")} value={city} onChange={manual(setCity)} required={requiredFields?.includes("city") ?? true} />
          <Field label={t("dash.earnings.seller_postal")} value={postal} onChange={manual(setPostal)} required={requiredFields?.includes("postal_code") ?? true} />
          <div className="space-y-1.5">
            <Label htmlFor={taxKindId} className="text-xs text-text-secondary">{t("dash.earnings.seller_tax_kind")}</Label>
            <select
              id={taxKindId}
              required
              value={kind}
              onChange={(event) => setKind(event.target.value)}
              className="flex h-10 w-full rounded-md border border-border bg-background px-3 text-sm"
            >
              <option value="">Select tax identifier type</option>
              <option value="bn">{t("dash.earnings.seller_tax_bn")}</option>
              <option value="sin">{t("dash.earnings.seller_tax_sin")}</option>
              <option value="other">{t("dash.earnings.seller_tax_other")}</option>
            </select>
          </div>
          <Field
            label={t("dash.earnings.seller_tax_id")}
            required={!provider?.tax_id_last4}
            value={taxId}
            onChange={setTaxId}
            autoComplete="off"
            hint={provider?.tax_id_last4 ? t("dash.earnings.seller_tax_id_hint") : undefined}
          />
          <Field
            label={t("dash.earnings.seller_gst")}
            required={false}
            value={gst}
            onChange={setGst}
            hint={t("dash.earnings.seller_gst_hint")}
          />
        </div>
        <label className="flex items-start gap-2 text-xs text-text-secondary">
          <input
            required
            type="checkbox"
            className="mt-0.5"
            checked={accepted}
            onChange={(event) => setAccepted(event.target.checked)}
          />
          <span>{t("dash.earnings.seller_terms")}</span>
        </label>
        {error && <p role="alert" className="rounded-lg border border-red-400/25 p-3 text-sm text-red-400">{error}</p>}
        <Button size="sm" type="submit" disabled={saving || !accepted || !country || requiredFields === null}>
          {saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null}
          {t("dash.earnings.seller_save")}
        </Button>
      </form></CardContent>
    </Card>
  );
}

function Field({
  label,
  value,
  onChange,
  hint,
  placeholder,
  maxLength,
  required = true,
  autoComplete,
  className,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  hint?: string;
  placeholder?: string;
  maxLength?: number;
  required?: boolean;
  autoComplete?: string;
  className?: string;
}) {
  const id = useId();
  return (
    <div className={`space-y-1.5 ${className || ""}`}>
      <Label htmlFor={id} className="text-xs text-text-secondary">{label}</Label>
      <Input
        id={id}
        required={required}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={placeholder}
        maxLength={maxLength}
        autoComplete={autoComplete}
      />
      {hint && <p className="text-[11px] leading-relaxed text-text-muted">{hint}</p>}
    </div>
  );
}
