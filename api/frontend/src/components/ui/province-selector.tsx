"use client";

import { cn } from "@/lib/utils";
import { Input, Label, Select } from "@/components/ui/input";
import { useEffect, useId, useRef, useState } from "react";
import { apiFetch, updateProfile } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useLocale } from "@/lib/locale";
import { Button } from "@/components/ui/button";

export const PROVINCES = [
  { code: "all", name: "All Provinces" },
  { code: "AB", name: "Alberta" },
  { code: "BC", name: "British Columbia" },
  { code: "MB", name: "Manitoba" },
  { code: "NB", name: "New Brunswick" },
  { code: "NL", name: "Newfoundland and Labrador" },
  { code: "NS", name: "Nova Scotia" },
  { code: "NT", name: "Northwest Territories" },
  { code: "NU", name: "Nunavut" },
  { code: "ON", name: "Ontario" },
  { code: "PE", name: "Prince Edward Island" },
  { code: "QC", name: "Quebec" },
  { code: "SK", name: "Saskatchewan" },
  { code: "YT", name: "Yukon" },
] as const;

interface ProvinceSelectorProps {
  value: string;
  onChange: (province: string) => void;
  showAll?: boolean;
  className?: string;
}

export function ProvinceSelector({ value, onChange, showAll = true, className }: ProvinceSelectorProps) {
  const options = showAll ? PROVINCES : PROVINCES.filter((p) => p.code !== "all");

  return (
    <Select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className={cn(className)}
    >
      {options.map((p) => (
        <option key={p.code} value={p.code}>
          {p.name}
        </option>
      ))}
    </Select>
  );
}


export type AddressLabels = { subdivision: string; city: string; postal_code: string };
type LocationCatalog = {
  countries: { code: string; name: string }[];
  subdivisions: { code: string; name: string; region: string }[];
  requires_subdivision: boolean;
  subdivision_mode?: "required" | "optional" | "none";
  labels?: AddressLabels;
  required_address_fields: string[];
};

/**
 * Choices come from the same ISO catalog the API uses to validate saves.
 * `purpose="address"` follows each country's postal format: the state field is
 * hidden where the post never uses one (the UK, Germany, France…), optional where
 * it may, and named the local way ("Prefecture", "County"). A host keeps its
 * subdivision, which is its scheduling region.
 */
export function LocationFields({ country, province, onCountryChange, onProvinceChange, onRequirementsChange, onLabelsChange, purpose = "host" }: {
  country: string; province: string;
  onCountryChange: (value: string) => void;
  onProvinceChange: (value: string) => void;
  onRequirementsChange?: (value: string[] | null) => void;
  onLabelsChange?: (value: AddressLabels | null) => void;
  purpose?: "host" | "address";
}) {
  const id = useId();
  const [countries, setCountries] = useState<LocationCatalog["countries"]>([]);
  const [catalog, setCatalog] = useState<LocationCatalog | null>(null);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  const [edited, setEdited] = useState<{ country?: boolean; province?: boolean }>({});
  const [touched, setTouched] = useState<{ country?: boolean; province?: boolean }>({});
  const selectedCountry = countries.some((item) => item.code === country) ? country : "";
  useEffect(() => {
    let active = true;
    setError("");
    apiFetch<LocationCatalog>("/api/locations")
      .then((result) => { if (active) setCountries(result.countries); })
      .catch((err) => { if (active) setError(err instanceof Error ? err.message : "Unable to load countries"); });
    return () => { active = false; };
  }, [retry]);
  useEffect(() => {
    let active = true;
    setCatalog(null);
    onRequirementsChange?.(null);
    onLabelsChange?.(null);
    if (!selectedCountry) return () => { active = false; };
    setError("");
    const query = `country=${encodeURIComponent(selectedCountry)}${purpose === "address" ? "&purpose=address" : ""}`;
    apiFetch<LocationCatalog>(`/api/locations?${query}`)
      .then((result) => {
        if (active) {
          setCatalog(result);
          onLabelsChange?.(result.labels ?? null);
          const required = result.requires_subdivision && !result.required_address_fields.includes("province")
            ? ["province", ...result.required_address_fields]
            : result.required_address_fields;
          onRequirementsChange?.(required);
        }
      })
      .catch((err) => { if (active) setError(err instanceof Error ? err.message : "Unable to load states and provinces"); });
    return () => { active = false; };
  }, [selectedCountry, retry, onRequirementsChange, onLabelsChange, purpose]);
  const selectedProvince = catalog?.subdivisions.some((item) => item.code === province) ? province : "";
  const provinceRequired = catalog?.requires_subdivision ?? true;
  const provinceLabel = catalog?.labels?.subdivision ?? "State, province, or territory";
  const provinceOptional = catalog?.subdivision_mode === "optional";
  // Every country's regions are offered. Only a country with none has nothing to pick.
  const provinceUnavailable = Boolean(catalog && catalog.subdivisions.length === 0);
  const countryInvalid = Boolean(edited.country && touched.country && !selectedCountry);
  const provinceInvalid = Boolean(edited.province && touched.province && provinceRequired && !selectedProvince);
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      <div className="space-y-1.5">
        <Label htmlFor={`${id}-country`}>Country</Label>
        <Select id={`${id}-country`} autoComplete="country" required value={selectedCountry}
          aria-invalid={countryInvalid ? "true" : "false"}
          aria-describedby={countryInvalid ? `${id}-country-error` : undefined}
          className={cn(countryInvalid && "border-red-400/60 focus:ring-red-400/40")}
          onBlur={() => { if (edited.country) setTouched((prev) => (prev.country ? prev : { ...prev, country: true })); }}
          onChange={(event) => {
            setEdited((prev) => ({ ...prev, country: true, province: false }));
            setTouched((prev) => ({ ...prev, province: false }));
            onCountryChange(event.target.value);
            onProvinceChange("");
          }}>
          <option value="">{countries.length ? "Select country" : "Loading countries…"}</option>
          {countries.map((item) => <option key={item.code} value={item.code}>{item.name}</option>)}
        </Select>
        {countryInvalid && <p id={`${id}-country-error`} className="text-xs text-red-400">Select a country.</p>}
      </div>
      <div className="space-y-1.5">
        <Label htmlFor={`${id}-province`}>{provinceLabel}{provinceOptional ? " (optional)" : ""}</Label>
        <Select id={`${id}-province`} autoComplete="address-level1" required={provinceRequired}
          value={selectedProvince} disabled={provinceUnavailable}
          aria-invalid={provinceInvalid ? "true" : "false"}
          aria-describedby={provinceInvalid ? `${id}-province-error` : undefined}
          className={cn(provinceInvalid && "border-red-400/60 focus:ring-red-400/40")}
          onBlur={() => { if (edited.province) setTouched((prev) => (prev.province ? prev : { ...prev, province: true })); }}
          onChange={(event) => {
            setEdited((prev) => (prev.province ? prev : { ...prev, province: true }));
            onProvinceChange(event.target.value);
          }}>
          <option value="">{provinceUnavailable ? "Not applicable" : provinceOptional ? "None" : `Select ${provinceLabel.toLowerCase()}`}</option>
          {catalog?.subdivisions.map((item) => <option key={item.code} value={item.code}>{item.name}</option>)}
        </Select>
        {provinceInvalid && <p id={`${id}-province-error`} className="text-xs text-red-400">Select a {provinceLabel.toLowerCase()}.</p>}
      </div>
      {error && <div role="alert" className="text-sm text-red-400 sm:col-span-2">
        {error} <Button type="button" variant="outline" size="sm" onClick={() => setRetry((value) => value + 1)}>Retry</Button>
      </div>}
    </div>
  );
}


type CapturedAddress = {
  address_line1: string; address_line2: string; city: string;
  postal_code: string; country: string; province: string;
};
type AddressSuggestion = { id: string; text: string; description: string; address: CapturedAddress };
type AddressReview = {
  address: CapturedAddress;
  errors: { field: string; message: string }[];
  warnings: { field: string; code: string; message: string }[];
  lines: string[];
};

/** The free check before a save. It only helps: the save applies the same rules, so a failed check never blocks. */
async function checkAddress(address: CapturedAddress): Promise<AddressReview | null> {
  try {
    const result = await apiFetch<AddressReview>("/api/locations/check", { method: "POST", body: JSON.stringify(address) });
    return result?.address && Array.isArray(result.errors) && Array.isArray(result.warnings) && Array.isArray(result.lines)
      ? result : null;
  } catch { return null; }
}

/** Bold the parts of a suggestion that match what was typed, as Google Maps does. */
function Highlight({ text, query }: { text: string; query: string }) {
  const lower = text.toLowerCase();
  const words = query.toLowerCase().split(/[\s,]+/).filter((word) => word.length > 1);
  if (!words.length || lower.length !== text.length) return <>{text}</>;
  const bold = new Array<boolean>(text.length).fill(false);
  for (const word of words) {
    for (let at = lower.indexOf(word); at !== -1; at = lower.indexOf(word, at + word.length)) {
      bold.fill(true, at, at + word.length);
    }
  }
  const parts: { text: string; bold: boolean }[] = [];
  for (let index = 0; index < text.length; index++) {
    const last = parts[parts.length - 1];
    if (last && last.bold === bold[index]) last.text += text[index];
    else parts.push({ text: text[index], bold: bold[index] });
  }
  // Plain stretches stay text nodes, so spaces survive in what a screen reader hears.
  return <>{parts.map((part, index) => part.bold
    ? <strong key={index} className="font-semibold">{part.text}</strong>
    : part.text)}</>;
}

/** Search helps fill the form; its fields remain editable and independently validated. */
export function AddressLookup({ country, onSelect, editRevision = 0 }: {
  country: string; onSelect: (address: CapturedAddress) => void; editRevision?: number;
}) {
  const id = useId();
  const { locale } = useLocale();
  const [query, setQuery] = useState("");
  const [suggestions, setSuggestions] = useState<AddressSuggestion[]>([]);
  const [activeIndex, setActiveIndex] = useState(-1);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [dismissed, setDismissed] = useState(false);
  // A search that finds nothing is not a dead end: a new street may simply not be listed yet.
  const [noMatches, setNoMatches] = useState(false);
  const requestVersion = useRef(0);
  useEffect(() => () => { requestVersion.current++; }, []);
  useEffect(() => {
    requestVersion.current++;
    setQuery(""); setSuggestions([]); setError(""); setDismissed(false); setNoMatches(false);
  }, [country, editRevision]);
  useEffect(() => {
    const version = ++requestVersion.current;
    setError(""); setNoMatches(false);
    if (query.trim().length < 3 || dismissed) {
      setSuggestions([]); setActiveIndex(-1); setLoading(false);
      return;
    }
    // The last results stay on screen until the next ones arrive, so the list
    // does not blink shut on every keystroke.
    const timer = setTimeout(async () => {
      setLoading(true);
      try {
        const result = await apiFetch<{ suggestions: AddressSuggestion[] }>("/api/locations/search", {
          method: "POST", body: JSON.stringify({ country, text: query, lang: locale }),
        });
        if (requestVersion.current === version) {
          setSuggestions(result.suggestions);
          setActiveIndex(-1);
          setNoMatches(result.suggestions.length === 0);
        }
      } catch (err) {
        if (requestVersion.current === version) {
          setSuggestions([]);
          setError(err instanceof Error ? err.message : "Address search is unavailable. Enter your address below.");
        }
      } finally { if (requestVersion.current === version) setLoading(false); }
    }, 250);
    return () => { clearTimeout(timer); if (requestVersion.current === version) requestVersion.current++; };
  }, [country, query, dismissed, locale]);
  // Each suggestion already holds its whole address, so picking one makes no further request.
  function choose(item: AddressSuggestion) {
    requestVersion.current++;
    onSelect(item.address); setQuery(""); setSuggestions([]); setLoading(false); setDismissed(true);
  }
  const open = suggestions.length > 0 && !dismissed;
  return <div className="relative space-y-1.5">
    <Label htmlFor={`${id}-search`}>Search address</Label>
    <div className="relative">
      <Input id={`${id}-search`} role="combobox" aria-autocomplete="list" aria-expanded={open} aria-busy={loading}
        aria-controls={`${id}-results`} aria-activedescendant={activeIndex >= 0 ? `${id}-result-${activeIndex}` : undefined}
        autoComplete="off" value={query} placeholder="Start typing your address"
        onChange={(event) => { requestVersion.current++; setQuery(event.target.value); setDismissed(false); }}
        onBlur={() => { requestVersion.current++; setDismissed(true); setLoading(false); }}
        onFocus={() => setDismissed(false)}
        onKeyDown={(event) => {
          if (event.key === "Escape") { requestVersion.current++; setDismissed(true); setLoading(false); }
          if (open && ["ArrowDown", "ArrowUp", "Enter"].includes(event.key)) {
            event.preventDefault();
            if (event.key === "Enter" && activeIndex >= 0) choose(suggestions[activeIndex]);
            else setActiveIndex((value) => (value + (event.key === "ArrowUp" ? -1 : 1) + suggestions.length) % suggestions.length);
          }
        }} />
      {open && <ul id={`${id}-results`} role="listbox" aria-label="Address suggestions"
        className="absolute left-0 right-0 top-full z-50 mt-1.5 max-h-72 w-full overflow-y-auto rounded-xl border border-border-light bg-[var(--popover-solid)] p-1.5 shadow-2xl">
        {suggestions.map((item, index) => <li key={item.id} id={`${id}-result-${index}`} role="option"
          aria-selected={activeIndex === index}
          className={cn(
            "cursor-pointer rounded-lg px-3 py-2.5 text-sm transition-colors hover:bg-[var(--popover-solid-2)]",
            activeIndex === index && "bg-[var(--popover-solid-2)]",
          )}
          onMouseEnter={() => setActiveIndex(index)}
          onMouseDown={(event) => event.preventDefault()} onClick={() => choose(item)}>
          <span className="block text-text-primary"><Highlight text={item.text} query={query} /></span>{" "}
          <span className="block text-xs text-text-secondary">{item.description}</span>
        </li>)}
      </ul>}
    </div>
    {loading && <p role="status" className="sr-only">Searching…</p>}
    {error && <p role="status" className="text-xs text-text-secondary">{error}</p>}
    {noMatches && !dismissed && <p role="status" className="text-xs text-text-secondary">
      No matching address found. That&apos;s fine: enter yours in the fields below.
    </p>}
    <p className="text-xs text-text-muted">Search for your address, or enter it below.</p>
    {query && <Button type="button" variant="outline" size="sm" onClick={() => {
      requestVersion.current++; setDismissed(true); setSuggestions([]); setLoading(false); setError("");
    }}>Enter manually</Button>}
  </div>;
}

export function AccountProfileForm({ step = "address", onSaved, settings = false }: {
  step?: "account" | "address"; onSaved?: () => void | Promise<void>; settings?: boolean;
}) {
  const { user, refreshUser } = useAuth();
  const [name, setName] = useState(user?.name || "");
  const [country, setCountry] = useState(user?.country || "");
  const [province, setProvince] = useState(user?.province || "");
  const [line1, setLine1] = useState(user?.address_line1 || "");
  const [line2, setLine2] = useState(user?.address_line2 || "");
  const [city, setCity] = useState(user?.city || "");
  const [postal, setPostal] = useState(user?.postal_code || "");
  const [accountType, setAccountType] = useState(user?.account_type || "");
  const [intent, setIntent] = useState(user?.onboarding_intent || "");
  const [businessName, setBusinessName] = useState(user?.business_name || "");
  const [gst, setGst] = useState(user?.gst_hst_number || "");
  const [requiredFields, setRequiredFields] = useState<string[] | null>(null);
  const [labels, setLabels] = useState<AddressLabels | null>(null);
  const cityLabel = labels?.city ?? "City or locality";
  const postalLabel = labels?.postal_code ?? "Postal code";
  const [edited, setEdited] = useState<Record<string, boolean>>({});
  const [touched, setTouched] = useState<Record<string, boolean>>({});
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [saved, setSaved] = useState(false);
  const [editRevision, setEditRevision] = useState(0);
  // A changed address is shown back in postal format before it is saved. Search
  // suggestions are OpenStreetMap's, not the post office's, so they get the same look.
  const [review, setReview] = useState<AddressReview | null>(null);
  const [savedSnapshot, setSavedSnapshot] = useState(() => ({
    name: (user?.name || "").trim(),
    account_type: user?.account_type || "",
    onboarding_intent: user?.onboarding_intent || "",
    business_name: user?.account_type === "business" ? (user?.business_name || "").trim() : "",
    gst_hst_number: user?.account_type === "business" ? (user?.gst_hst_number || "").trim() : "",
    country: user?.country || "",
    province: user?.province || "",
    address_line1: (user?.address_line1 || "").trim(),
    address_line2: (user?.address_line2 || "").trim(),
    city: (user?.city || "").trim(),
    postal_code: (user?.postal_code || "").trim(),
  }));
  const manual = (set: (value: string) => void, value: string) => {
    setEditRevision((revision) => revision + 1); setReview(null); set(value);
  };
  const markEdited = (field: string) => setEdited((prev) => (prev[field] ? prev : { ...prev, [field]: true }));
  const touch = (field: string) => setTouched((prev) => (prev[field] ? prev : { ...prev, [field]: true }));
  const blurField = (field: string) => { if (edited[field]) touch(field); };
  const id = useId();
  const accountFields = step === "account" || settings;
  const addressFields = step === "address" || settings;

  const nameRequired = !user?.name?.trim();
  const nameValid = !nameRequired || Boolean(name.trim());
  const intentValid = !settings || Boolean(intent.trim());
  const typeValid = accountType === "individual" || accountType === "business";
  const businessValid = accountType !== "business" || Boolean(businessName.trim());
  const accountValid = !accountFields || (nameValid && intentValid && typeValid && businessValid);

  const line1Required = requiredFields?.includes("address_line1") ?? true;
  const cityRequired = requiredFields?.includes("city") ?? true;
  const postalRequired = requiredFields?.includes("postal_code") ?? true;
  const addressFieldValues: Record<string, string> = {
    province,
    address_line1: line1,
    city,
    postal_code: postal,
  };
  const addressValid =
    !addressFields ||
    (Boolean(country.trim()) &&
      requiredFields !== null &&
      requiredFields.every((field) => Boolean(addressFieldValues[field]?.trim())));

  const nameInvalid = Boolean(accountFields && nameRequired && edited.name && touched.name && !nameValid);
  const typeInvalid = Boolean(accountFields && edited.accountType && touched.accountType && !typeValid);
  const businessInvalid = Boolean(accountFields && accountType === "business" && edited.businessName && touched.businessName && !businessValid);
  const line1Invalid = Boolean(addressFields && line1Required && edited.line1 && touched.line1 && !line1.trim());
  const cityInvalid = Boolean(addressFields && cityRequired && edited.city && touched.city && !city.trim());
  const postalInvalid = Boolean(addressFields && postalRequired && edited.postal && touched.postal && !postal.trim());
  const formValid = accountValid && addressValid;
  const effectiveBusinessName = accountType === "business" ? businessName.trim() : "";
  const effectiveGst = accountType === "business" ? gst.trim() : "";
  const accountDirty =
    (!user?.name?.trim() && name.trim() !== savedSnapshot.name) ||
    accountType !== savedSnapshot.account_type ||
    (settings && intent !== savedSnapshot.onboarding_intent) ||
    effectiveBusinessName !== savedSnapshot.business_name ||
    effectiveGst !== savedSnapshot.gst_hst_number;
  const addressDirty =
    country !== savedSnapshot.country ||
    province !== savedSnapshot.province ||
    line1.trim() !== savedSnapshot.address_line1 ||
    line2.trim() !== savedSnapshot.address_line2 ||
    city.trim() !== savedSnapshot.city ||
    postal.trim() !== savedSnapshot.postal_code;
  const isDirty = (accountFields && accountDirty) || (addressFields && addressDirty);

  async function save(address: CapturedAddress) {
    setSaving(true); setError(""); setSaved(false);
    try {
      await updateProfile({
        ...(accountFields ? { ...(!user?.name?.trim() ? { name: name.trim() } : {}),
          account_type: accountType,
          ...(settings ? { onboarding_intent: intent } : {}),
          business_name: effectiveBusinessName,
          gst_hst_number: effectiveGst } : {}),
        ...(addressFields ? address : {}),
      });
      if (addressFields) {
        // Show what was stored: the cleaned-up address, not what was typed.
        setCountry(address.country); setProvince(address.province); setLine1(address.address_line1);
        setLine2(address.address_line2); setCity(address.city); setPostal(address.postal_code);
      }
      setReview(null);
      setSavedSnapshot({
        name: (!user?.name?.trim() ? name : user?.name || "").trim(),
        account_type: accountType,
        onboarding_intent: intent,
        business_name: effectiveBusinessName,
        gst_hst_number: effectiveGst,
        country: address.country,
        province: address.province,
        address_line1: address.address_line1.trim(),
        address_line2: address.address_line2.trim(),
        city: address.city.trim(),
        postal_code: address.postal_code.trim(),
      });
      setEdited({});
      setTouched({});
      await refreshUser({ throwOnError: true });
      await onSaved?.();
      if (settings) setSaved(true);
    } catch (err) { setError(err instanceof Error ? err.message : "Unable to save your profile"); }
    finally { setSaving(false); }
  }

  return (
    <form noValidate className="space-y-5" onSubmit={async (event) => {
      event.preventDefault();
      if (!formValid) {
        setEdited({ name: true, accountType: true, businessName: true, line1: true, city: true, postal: true });
        setTouched({ name: true, accountType: true, businessName: true, line1: true, city: true, postal: true });
        event.currentTarget.querySelector<HTMLElement>('[aria-invalid="true"], :invalid')?.focus();
        return;
      }
      if (!isDirty) return;
      const typed: CapturedAddress = { country, province, address_line1: line1, address_line2: line2, city, postal_code: postal };
      if (addressFields && addressDirty) {
        setSaving(true); setError(""); setSaved(false);
        const checked = await checkAddress(typed);
        setSaving(false);
        if (checked) {
          if (checked.errors.length) { setError(checked.errors.map((item) => item.message).join(" ")); return; }
          setReview(checked);
          return;
        }
      }
      await save(typed);
    }}>
      {accountFields && <>
        {!user?.name?.trim() && <div className="space-y-1.5"><Label htmlFor={`${id}-name`}>Name</Label>
          <Input id={`${id}-name`} autoComplete="name" value={name} required maxLength={64}
            aria-invalid={nameInvalid ? "true" : "false"}
            aria-describedby={nameInvalid ? `${id}-name-error` : undefined}
            className={cn(nameInvalid && "border-red-400/60 focus:ring-red-400/40")}
            onBlur={() => blurField("name")}
            onChange={(event) => { markEdited("name"); setName(event.target.value); }} />
          {nameInvalid && <p id={`${id}-name-error`} className="text-xs text-red-400">Enter your name.</p>}
        </div>}
        {settings && <fieldset className="space-y-2"><legend className="text-sm font-medium">How will you use Xcelsior?</legend>
          <p className="text-sm text-text-secondary">You can switch between renting and providing here anytime. Payout setup stays on Earnings.</p>
          {[["rent", "Rent GPUs"], ["provide", "Provide GPUs"], ["both", "Rent and provide GPUs"]].map(([value, label]) =>
            <label key={value} className={cn("flex cursor-pointer items-center gap-3 rounded-xl border p-4 text-sm transition-colors hover:bg-surface-hover", intent === value ? "border-accent-cyan/40 bg-accent-cyan/5" : "border-border bg-navy/30")}>
              <input type="radio" name={`${id}-intent`} value={value} checked={intent === value} required onChange={() => setIntent(value)} className="h-4 w-4 accent-sky-500" /><span className="font-medium">{label}</span>
            </label>)}
        </fieldset>}
        <div className="space-y-1.5"><Label htmlFor={`${id}-type`}>Account type</Label>
          <Select id={`${id}-type`} data-field="account-type" required value={accountType}
            aria-invalid={typeInvalid ? "true" : "false"}
            aria-describedby={typeInvalid ? `${id}-type-error` : undefined}
            className={cn(typeInvalid && "border-red-400/60 focus:ring-red-400/40")}
            onBlur={() => blurField("accountType")}
            onChange={(event) => {
              markEdited("accountType");
              setEdited((prev) => ({ ...prev, businessName: false }));
              setTouched((prev) => ({ ...prev, businessName: false }));
              setAccountType(event.target.value);
            }}>
            <option value="">Select account type</option><option value="individual">Personal</option><option value="business">Business</option>
          </Select>
          {typeInvalid && <p id={`${id}-type-error`} className="text-xs text-red-400">Select an account type.</p>}
          {accountType === "individual" && <p className="text-sm text-text-secondary">A personal account does not need a business name, even if you provide GPUs.</p>}
        </div>
        {accountType === "business" && <>
          <div className="space-y-1.5"><Label htmlFor={`${id}-business`}>Legal business name</Label>
            <Input id={`${id}-business`} autoComplete="organization" required maxLength={200} value={businessName}
              aria-invalid={businessInvalid ? "true" : "false"}
              aria-describedby={businessInvalid ? `${id}-business-error` : undefined}
              className={cn(businessInvalid && "border-red-400/60 focus:ring-red-400/40")}
              onBlur={() => blurField("businessName")}
              onChange={(event) => { markEdited("businessName"); setBusinessName(event.target.value); }} />
            {businessInvalid && <p id={`${id}-business-error`} className="text-xs text-red-400">Enter your legal business name.</p>}
          </div>
          <div className="space-y-1.5"><Label htmlFor={`${id}-gst`}>GST/HST registration number (optional)</Label>
            <Input id={`${id}-gst`} maxLength={20} value={gst} onChange={(event) => setGst(event.target.value)} /></div>
        </>}
      </>}
      {addressFields && <>
        <AddressLookup country={country} editRevision={editRevision} onSelect={(address) => {
          setLine1(address.address_line1); setLine2(address.address_line2); setCity(address.city);
          setPostal(address.postal_code); setCountry(address.country); setProvince(address.province);
          setTouched((prev) => ({ ...prev, line1: false, city: false, postal: false }));
          setReview(null);
        }} />
        <LocationFields country={country} province={province} purpose="address" onLabelsChange={setLabels}
          onCountryChange={(value) => { setReview(null); setCountry(value); }}
          onProvinceChange={(value) => manual(setProvince, value)} onRequirementsChange={setRequiredFields} />
        <div className="space-y-1.5"><Label htmlFor={`${id}-line1`}>Street address</Label>
          <Input id={`${id}-line1`} autoComplete="address-line1" required={line1Required} maxLength={200} value={line1}
            aria-invalid={line1Invalid ? "true" : "false"}
            aria-describedby={line1Invalid ? `${id}-line1-error` : undefined}
            className={cn(line1Invalid && "border-red-400/60 focus:ring-red-400/40")}
            onBlur={() => blurField("line1")}
            onChange={(event) => { markEdited("line1"); manual(setLine1, event.target.value); }} />
          {line1Invalid && <p id={`${id}-line1-error`} className="text-xs text-red-400">Enter your street address.</p>}
        </div>
        <div className="space-y-1.5"><Label htmlFor={`${id}-line2`}>Apartment, suite, or unit (optional)</Label>
          <Input id={`${id}-line2`} autoComplete="address-line2" maxLength={200} value={line2} onChange={(event) => manual(setLine2, event.target.value)} /></div>
        <div className="grid gap-3 sm:grid-cols-2">
          <div className="space-y-1.5"><Label htmlFor={`${id}-city`}>{cityLabel}</Label>
            <Input id={`${id}-city`} autoComplete="address-level2" required={cityRequired} maxLength={80} value={city}
              aria-invalid={cityInvalid ? "true" : "false"}
              aria-describedby={cityInvalid ? `${id}-city-error` : undefined}
              className={cn(cityInvalid && "border-red-400/60 focus:ring-red-400/40")}
              onBlur={() => blurField("city")}
              onChange={(event) => { markEdited("city"); manual(setCity, event.target.value); }} />
            {cityInvalid && <p id={`${id}-city-error`} className="text-xs text-red-400">Enter your {cityLabel.toLowerCase()}.</p>}
          </div>
          <div className="space-y-1.5"><Label htmlFor={`${id}-postal`}>{postalLabel}</Label>
            <Input id={`${id}-postal`} autoComplete="postal-code" required={postalRequired} maxLength={32} value={postal}
              aria-invalid={postalInvalid ? "true" : "false"}
              aria-describedby={postalInvalid ? `${id}-postal-error` : undefined}
              className={cn(postalInvalid && "border-red-400/60 focus:ring-red-400/40")}
              onBlur={() => blurField("postal")}
              onChange={(event) => { markEdited("postal"); manual(setPostal, event.target.value); }} />
            {postalInvalid && <p id={`${id}-postal-error`} className="text-xs text-red-400">Enter your {postalLabel === "Postal code" ? "postal code" : postalLabel}.</p>}
          </div>
        </div>
      </>}
      {error && <p role="alert" className="rounded-lg border border-red-400/25 bg-red-400/5 p-3 text-sm text-red-400">{error}</p>}
      {saved && <p role="status" className="text-sm text-emerald">Profile saved.</p>}
      {review ? <div role="group" aria-labelledby={`${id}-review-title`} className="space-y-3 rounded-xl border border-border bg-navy/30 p-4">
        <p id={`${id}-review-title`} className="text-sm font-medium text-text-primary">Is this address right?</p>
        <address className="text-sm not-italic leading-relaxed text-text-primary">
          {review.lines.map((line, index) => <span key={index} className="block">{line}</span>)}
        </address>
        {review.warnings.map((warning) => <p key={warning.code} className="text-sm text-accent-gold">{warning.message}</p>)}
        <div className="flex flex-wrap gap-2">
          <Button type="button" disabled={saving} onClick={() => void save(review.address)}>
            {saving ? "Saving…" : settings ? "Yes, save this address" : "Yes, continue"}
          </Button>
          <Button type="button" variant="outline" disabled={saving} onClick={() => {
            setReview(null); document.getElementById(`${id}-line1`)?.focus();
          }}>Edit address</Button>
        </div>
      </div>
      : <Button className={settings ? "" : "w-full"} type="submit" disabled={saving || !formValid || !isDirty}>
        {saving ? "Saving…" : settings ? "Save profile" : "Continue"}
      </Button>}
    </form>
  );
}


export function HostLocationForm({ host, onSaved }: {
  host: import("@/lib/api").Host; onSaved: (host: import("@/lib/api").Host) => void;
}) {
  const { user } = useAuth();
  // A host without a confirmed site starts from the owner's address; the owner
  // still saves it, because a machine can sit somewhere else.
  const prefill = host.location_complete === false && !host.province && user?.country
    ? { country: user.country, province: user.province || "" } : null;
  const [country, setCountry] = useState(prefill?.country ?? host.country ?? "");
  const [province, setProvince] = useState(prefill?.province ?? host.province ?? "");
  const [requirements, setRequirements] = useState<string[] | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [saved, setSaved] = useState(false);
  const locationValid = Boolean(country.trim()) && requirements !== null && (!requirements.includes("province") || Boolean(province.trim()));
  const isDirty = host.location_complete === false || country !== (host.country ?? "") || province !== (host.province ?? "");
  return <form noValidate className="space-y-4 rounded-xl border border-border bg-surface p-5" onSubmit={async (event) => {
    event.preventDefault();
    if (!locationValid) return;
    setSaving(true); setError(""); setSaved(false);
    try {
      if (typeof host.version !== "number") throw new Error("Reload this host's details before editing its location.");
      const result = await apiFetch<{ host: import("@/lib/api").Host }>(`/api/hosts/${encodeURIComponent(host.host_id)}/location`, {
        method: "PATCH", body: JSON.stringify({ country, province, expected_version: host.version }),
      });
      onSaved({ ...host, ...result.host }); setSaved(true);
    } catch (err) { setError(err instanceof Error ? err.message : "Unable to save the host location"); }
    finally { setSaving(false); }
  }}>
    <div><h2 className="text-base font-semibold">Host location</h2>
      <p className="mt-1 text-sm text-text-secondary">Where this machine is located. Update it here if the host moves.</p>
      {host.location_complete === false && <p className="mt-2 text-sm text-accent-cyan">Complete this location before the host can appear in available GPU inventory.</p>}
      {prefill && <p className="mt-1 text-sm text-text-secondary">Filled in from your account address. Change it if this machine is somewhere else.</p>}
    </div>
    <LocationFields country={country} province={province}
      onCountryChange={(value) => { setCountry(value); setSaved(false); }}
      onProvinceChange={(value) => { setProvince(value); setSaved(false); }}
      onRequirementsChange={setRequirements} />
    {error && <p role="alert" className="text-sm text-red-400">{error}</p>}
    {saved && <p role="status" className="text-sm text-emerald">Host location saved.</p>}
    <Button type="submit" disabled={saving || !locationValid || !isDirty || typeof host.version !== "number"}>{saving ? "Saving…" : "Save location"}</Button>
  </form>;
}
