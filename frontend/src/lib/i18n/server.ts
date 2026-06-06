import { cookies } from "next/headers";
import type { Locale } from "@/lib/locale";
import enPublic from "@/lib/i18n/en-public";
import frPublic from "@/lib/i18n/fr-public";

const dictionaries = { en: enPublic, fr: frPublic } as const;

export async function getServerLocale(): Promise<Locale> {
  const jar = await cookies();
  const stored = jar.get("xcelsior-locale")?.value;
  return stored === "fr" ? "fr" : "en";
}

export function createTranslator(locale: Locale) {
  const dict = dictionaries[locale];
  return (key: string): string => dict[key] ?? dictionaries.en[key] ?? key;
}