import { ServerlessFeatureGate } from "@/features/serverless/feature-gate";

export default function InferenceLayout({ children }: { children: React.ReactNode }) {
  return <ServerlessFeatureGate>{children}</ServerlessFeatureGate>;
}