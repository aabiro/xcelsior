"use client";

import { useEffect, useState } from "react";
import { apiFetch } from "@/lib/api";

/**
 * A deploy region code ("ca-on") is a country and one of its states or
 * provinces. People pick "Canada", then "Ontario"; the code stays underneath.
 */
export function regionParts(code: string): { country: string; subdivision: string } {
  const match = /^([a-z]{2})-([a-z0-9]{1,3})$/i.exec(code.trim());
  if (!match) return { country: "", subdivision: code };
  return { country: match[1].toUpperCase(), subdivision: match[2].toUpperCase() };
}

export function countryName(country: string, locale: string): string {
  if (!country) return "";
  try {
    return new Intl.DisplayNames([locale, "en"], { type: "region" }).of(country) ?? country;
  } catch {
    return country;
  }
}

/** Region code ("ca-on") to its state or province name ("Ontario"). */
type Subdivisions = Record<string, string>;
const cache = new Map<string, Promise<Subdivisions>>();

/** State and province names from the same ISO catalog the API validates addresses with. */
function loadSubdivisions(country: string): Promise<Subdivisions> {
  let pending = cache.get(country);
  if (!pending) {
    pending = apiFetch<{ subdivisions?: { code: string; name: string; region?: string }[] }>(
      `/api/locations?country=${encodeURIComponent(country)}`,
    )
      .then((result) => Object.fromEntries((result.subdivisions ?? []).map((item) => [
        (item.region || `${country}-${item.code}`).toLowerCase(),
        item.name,
      ])))
      .catch(() => {
        cache.delete(country);
        return {};
      });
    cache.set(country, pending);
  }
  return pending;
}

/** Names for the countries behind `codes`; a code whose name has not loaded shows as its letters. */
export function useSubdivisionNames(codes: string[]): Subdivisions {
  const countries = [...new Set(codes.map((code) => regionParts(code).country).filter(Boolean))].sort();
  const key = countries.join(",");
  const [names, setNames] = useState<Subdivisions>({});
  useEffect(() => {
    let active = true;
    const wanted = key ? key.split(",") : [];
    Promise.all(wanted.map(loadSubdivisions)).then((maps) => {
      if (active) setNames(Object.assign({}, ...maps));
    });
    return () => { active = false; };
  }, [key]);
  return names;
}

export function subdivisionName(code: string, names: Subdivisions): string {
  return names[code.trim().toLowerCase()] ?? regionParts(code).subdivision;
}

export function regionLabel(code: string, locale: string, names: Subdivisions): string {
  const { country } = regionParts(code);
  if (!country) return code;
  return `${subdivisionName(code, names)}, ${countryName(country, locale)}`;
}
