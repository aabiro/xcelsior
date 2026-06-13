"use client";

import { useRouter } from "next/navigation";
import { LaunchInstanceModal } from "@/components/instances/launch-instance-modal";

export default function NewInstancePage() {
  const router = useRouter();

  return (
    <LaunchInstanceModal
      open
      onClose={() => router.push("/dashboard/instances")}
      onLaunched={() => { /* modal shows its own success step; user chooses where to go */ }}
    />
  );
}
