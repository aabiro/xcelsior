"use client";

import { useId } from "react";
import { PickerSelect } from "@/components/ui/picker-select";
import { useLocale } from "@/lib/locale";
import { countryName, regionParts, subdivisionName, useSubdivisionNames } from "./region-names";

type NativeSelect = Parameters<React.ComponentProps<typeof PickerSelect>["renderNative"]>[0];

interface RegionPickerProps {
  /** Region codes the chosen GPU runs in, from live inventory ("ca-on"). */
  regions: string[];
  value: string;
  onChange: (region: string) => void;
  /** The form's own native select, for wider screens. */
  renderNative: (select: NativeSelect) => React.ReactNode;
  /** Wraps each of the two fields: its label and its control. */
  field: (label: React.ReactNode, control: React.ReactNode, key: string) => React.ReactNode;
  labelClassName?: string;
  fieldClassName?: string;
}

/**
 * Where a deploy runs: the country first, then its state or province, by name.
 * The picker used to list the raw region code ("ca-on"); people choose "Canada",
 * then "Ontario", and the code stays underneath (owner review, 2026-11-23).
 * Both lists come only from the regions the chosen GPU is in.
 */
export function RegionPicker({ regions, value, onChange, renderNative, field, labelClassName, fieldClassName }: RegionPickerProps) {
  const { t, locale } = useLocale();
  const ids = useId();
  const names = useSubdivisionNames(regions);

  const countries = [...new Set(regions.map((code) => regionParts(code).country))];
  const country = regionParts(value).country || (value ? "" : countries[0] ?? "");
  const inCountry = regions.filter((code) => regionParts(code).country === country);

  const countryOptions = countries.map((code) => ({
    value: code,
    label: code ? countryName(code, locale) : t("dash.serverless.region_other"),
  }));
  const regionOptions = inCountry.map((code) => ({ value: code, label: subdivisionName(code, names) }));
  const empty = regions.length === 0;

  return (
    <>
      {field(
        <label id={`${ids}-country-label`} htmlFor={`${ids}-country`} className={labelClassName}>
          {t("dash.serverless.country")}
        </label>,
        <PickerSelect
          id={`${ids}-country`}
          aria-labelledby={`${ids}-country-label`}
          value={country}
          options={countryOptions}
          placeholder={t(empty ? "dash.serverless.region_pick_gpu" : "dash.serverless.country_pick")}
          onChange={(next) => {
            const first = regions.find((code) => regionParts(code).country === next);
            if (first) onChange(first);
          }}
          renderNative={renderNative}
          className={fieldClassName}
        />,
        "country",
      )}
      {field(
        <label id={`${ids}-region-label`} htmlFor={`${ids}-region`} className={labelClassName}>
          {t("dash.serverless.province")}
        </label>,
        <PickerSelect
          id={`${ids}-region`}
          aria-labelledby={`${ids}-region-label`}
          value={value}
          options={regionOptions}
          placeholder={t(empty ? "dash.serverless.region_pick_gpu" : "dash.serverless.province_pick")}
          onChange={onChange}
          renderNative={renderNative}
          className={fieldClassName}
        />,
        "region",
      )}
    </>
  );
}
