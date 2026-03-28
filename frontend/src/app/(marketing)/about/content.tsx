"use client";

import Link from "next/link";
import { ArrowRight, MapPin, Shield, Leaf, Users, Zap, Heart } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useLocale } from "@/lib/locale";

const valueKeys = [
  { icon: Shield, title: "about.val_sovereignty_title", desc: "about.val_sovereignty_desc" },
  { icon: Leaf, title: "about.val_green_title", desc: "about.val_green_desc" },
  { icon: Users, title: "about.val_community_title", desc: "about.val_community_desc" },
  { icon: Zap, title: "about.val_access_title", desc: "about.val_access_desc" },
  { icon: Heart, title: "about.val_canada_title", desc: "about.val_canada_desc" },
  { icon: MapPin, title: "about.val_local_title", desc: "about.val_local_desc" },
];

const milestoneKeys = [
  { year: "about.journey_2024_title", events: ["about.journey_2024_p1", "about.journey_2024_p2"] },
  { year: "about.journey_2025_title", events: ["about.journey_2025_p1", "about.journey_2025_p2", "about.journey_2025_p3"] },
  { year: "about.journey_2026_title", events: ["about.journey_2026_p1"] },
];

export function AboutContent() {
  const { t } = useLocale();

  return (
    <div className="mx-auto max-w-7xl px-6 py-24">
      {/* Hero */}
      <div className="text-center mb-20">
        <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-accent-red/30 bg-accent-red/10 px-4 py-1.5">
          <MapPin className="h-3.5 w-3.5 text-accent-red" />
          <span className="text-xs font-medium text-accent-red">
            {t("about.badge")}
          </span>
        </div>
        <h1 className="text-4xl font-bold md:text-5xl lg:text-6xl">
          {t("about.title").split(/(truly Canadian|véritablement canadien)/i).map((part, i) =>
            /truly Canadian|véritablement canadien/i.test(part) ? (
              <span key={i} className="bg-gradient-to-r from-accent-red to-accent-gold bg-clip-text text-transparent">
                {part}
              </span>
            ) : (
              <span key={i}>{part}</span>
            ),
          )}
        </h1>
        <p className="mt-6 text-lg text-text-secondary max-w-3xl mx-auto leading-relaxed">
          {t("about.subtitle")}
        </p>
      </div>

      {/* Mission */}
      <div className="rounded-xl border border-accent-gold/30 bg-accent-gold/5 p-8 md:p-12 mb-20">
        <h2 className="text-2xl font-bold mb-4">{t("about.mission_title")}</h2>
        <p className="text-lg text-text-secondary leading-relaxed">
          {t("about.mission_p1")}
        </p>
      </div>

      {/* Values */}
      <div className="mb-20">
        <h2 className="text-2xl font-bold text-center mb-12">{t("about.values_title")}</h2>
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {valueKeys.map((v) => (
            <div key={v.title} className="rounded-xl border border-border bg-surface p-6 card-hover">
              <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-lg bg-ice-blue/10">
                <v.icon className="h-5 w-5 text-ice-blue" />
              </div>
              <h3 className="mb-2 text-lg font-semibold">{t(v.title)}</h3>
              <p className="text-sm text-text-secondary leading-relaxed">{t(v.desc)}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Timeline */}
      <div className="mb-20">
        <h2 className="text-2xl font-bold text-center mb-12">{t("about.journey_title")}</h2>
        <div className="relative mx-auto max-w-2xl">
          <div className="absolute left-4 top-0 bottom-0 w-px bg-border" />
          <div className="space-y-8">
            {milestoneKeys.flatMap((group) =>
              group.events.map((eventKey, i) => (
                <div key={eventKey} className="relative pl-12">
                  <div className="absolute left-2.5 top-1.5 h-3 w-3 rounded-full border-2 border-accent-gold bg-navy" />
                  {i === 0 && <span className="text-xs font-mono text-accent-gold">{t(group.year)}</span>}
                  <p className="text-sm text-text-secondary mt-0.5">{t(eventKey)}</p>
                </div>
              )),
            )}
          </div>
        </div>
      </div>

      {/* CTA */}
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-4">{t("about.cta_title")}</h2>
        <p className="text-text-secondary mb-8">{t("about.cta_desc")}</p>
        <Link href="/register">
          <Button size="lg">
            {t("about.cta_button")} <ArrowRight className="h-4 w-4" />
          </Button>
        </Link>
      </div>
    </div>
  );
}
