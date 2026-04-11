"use client";

import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { LaunchInstanceForm } from "@/components/instances/launch-instance-form";
import { useLocale } from "@/lib/locale";

export default function NewInstancePage() {
  const { t } = useLocale();

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <div className="flex items-center gap-3">
        <Link href="/dashboard/instances">
          <Button variant="ghost" size="icon" className="h-8 w-8">
            <ArrowLeft className="h-4 w-4" />
          </Button>
        </Link>
        <div>
          <h1 className="text-2xl font-bold">{t("dash.newinstance.title")}</h1>
          <p className="text-sm text-text-secondary">{t("dash.newinstance.subtitle")}</p>
        </div>
      </div>

      <LaunchInstanceForm />
    </div>
  );
}
