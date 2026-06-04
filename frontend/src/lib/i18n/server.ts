import { cookies } from "next/headers";
import type { Locale } from "@/lib/locale";
import en from "@/lib/i18n/en";
import fr from "@/lib/i18n/fr";

const dictionaries = { en, fr } as const;

export async function getServerLocale(): Promise<Locale> {
  const jar = await cookies();
  const stored = jar.get("xcelsior-locale")?.value;
  return stored === "fr" ? "fr" : "en";
}

export function createTranslator(locale: Locale) {
  const dict = dictionaries[locale];
  return (key: string): string => dict[key] ?? dictionaries.en[key] ?? key;
}