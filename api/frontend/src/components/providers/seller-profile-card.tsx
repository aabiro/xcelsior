"use client";

import { useState } from "react";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input, Label } from "@/components/ui/input";
import * as api from "@/lib/api";
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

export function SellerProfileCard({
  provider,
  onSaved,
}: {
  provider: SellerFields | null;
  onSaved: () => void;
}) {
  const { t } = useLocale();
  const [legalName, setLegalName] = useState(provider?.legal_name || "");
  const [line1, setLine1] = useState(provider?.address_line1 || "");
  const [line2, setLine2] = useState(provider?.address_line2 || "");
  const [city, setCity] = useState(provider?.city || "");
  const [province, setProvince] = useState(provider?.province || "ON");
  const [postal, setPostal] = useState(provider?.postal_code || "");
  const [country, setCountry] = useState(provider?.country || "CA");
  const [kind, setKind] = useState(provider?.tax_id_kind || "bn");
  const [taxId, setTaxId] = useState("");
  const [gst, setGst] = useState(provider?.gst_hst_number || "");
  const [accepted, setAccepted] = useState(Boolean(provider?.seller_profile_complete));
  const [saving, setSaving] = useState(false);

  const save = async () => {
    setSaving(true);
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
      onSaved();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : t("dash.earnings.seller_save"));
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
      <CardContent className="space-y-3">
        {provider?.tax_id_last4 && (
          <p className="text-xs text-text-muted">
            {t("dash.earnings.seller_on_file")} {provider.tax_id_last4}
          </p>
        )}
        <div className="grid gap-3 sm:grid-cols-2">
          <Field label={t("dash.earnings.seller_legal_name")} value={legalName} onChange={setLegalName} />
          <Field label={t("dash.earnings.seller_country")} value={country} onChange={setCountry} maxLength={2} />
          <Field label={t("dash.earnings.seller_address")} value={line1} onChange={setLine1} className="sm:col-span-2" />
          <Field label={t("dash.earnings.seller_address2")} value={line2} onChange={setLine2} className="sm:col-span-2" />
          <Field label={t("dash.earnings.seller_city")} value={city} onChange={setCity} />
          <Field label={t("dash.earnings.seller_province")} value={province} onChange={setProvince} />
          <Field label={t("dash.earnings.seller_postal")} value={postal} onChange={setPostal} />
          <div className="space-y-1.5">
            <Label className="text-xs text-text-secondary">{t("dash.earnings.seller_tax_kind")}</Label>
            <select
              value={kind}
              onChange={(event) => setKind(event.target.value)}
              className="flex h-10 w-full rounded-md border border-border bg-background px-3 text-sm"
            >
              <option value="bn">{t("dash.earnings.seller_tax_bn")}</option>
              <option value="sin">{t("dash.earnings.seller_tax_sin")}</option>
              <option value="other">{t("dash.earnings.seller_tax_other")}</option>
            </select>
          </div>
          <Field
            label={t("dash.earnings.seller_tax_id")}
            value={taxId}
            onChange={setTaxId}
            autoComplete="off"
            hint={provider?.tax_id_last4 ? t("dash.earnings.seller_tax_id_hint") : undefined}
          />
          <Field
            label={t("dash.earnings.seller_gst")}
            value={gst}
            onChange={setGst}
            placeholder="123456789RT0001"
            hint={t("dash.earnings.seller_gst_hint")}
          />
        </div>
        <label className="flex items-start gap-2 text-xs text-text-secondary">
          <input
            type="checkbox"
            className="mt-0.5"
            checked={accepted}
            onChange={(event) => setAccepted(event.target.checked)}
          />
          <span>{t("dash.earnings.seller_terms")}</span>
        </label>
        <Button size="sm" onClick={save} disabled={saving || !accepted}>
          {saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null}
          {t("dash.earnings.seller_save")}
        </Button>
      </CardContent>
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
  autoComplete,
  className,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  hint?: string;
  placeholder?: string;
  maxLength?: number;
  autoComplete?: string;
  className?: string;
}) {
  return (
    <div className={`space-y-1.5 ${className || ""}`}>
      <Label className="text-xs text-text-secondary">{label}</Label>
      <Input
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
