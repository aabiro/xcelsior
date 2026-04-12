"use client";

import { useRouter } from "next/navigation";
import { LaunchInstanceModal } from "@/components/instances/launch-instance-modal";

export default function NewInstancePage() {
  const router = useRouter();

  return (
    <LaunchInstanceModal
      open
      onClose={() => router.push("/dashboard/instances")}
      onLaunched={() => router.push("/dashboard/instances")}
    />
  );
}
