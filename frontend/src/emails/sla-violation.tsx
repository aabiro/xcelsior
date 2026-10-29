import { Heading, Hr, Link, Section, Text } from "@react-email/components";
import { EmailLayout, baseStyles } from "./layout";

interface SlaViolationProps {
  name: string;
  hostId: string;
  hostname: string;
  metric: string;
  threshold: string;
  actual: string;
  timestamp: string;
  dashboardUrl?: string;
}

export default function SlaViolationEmail({
  name = "there",
  hostId = "host-3b9a1f",
  hostname = "gpu-node-07",
  metric = "Uptime",
  threshold = "99.9%",
  actual = "98.7%",
  timestamp = "2026-02-15T14:32:00Z",
  dashboardUrl = "https://xcelsior.ca/dashboard/hosts",
}: SlaViolationProps) {
  const formattedTime = new Date(timestamp).toLocaleString("en-CA", {
    dateStyle: "medium",
    timeStyle: "short",
    timeZone: "America/Toronto",
  });

  return (
    <EmailLayout preview={`SLA violation on ${hostname}: ${metric} below threshold`}>
      <Heading style={baseStyles.heading}>SLA Violation Detected</Heading>
      <Text style={baseStyles.text}>
        Hi {name}, an SLA threshold has been breached on one of your hosts.
      </Text>

      <Section
        style={{
          ...baseStyles.card,
          borderColor: "#dc2626",
        }}
      >
        <Text style={{ ...baseStyles.text, margin: "0 0 8px" }}>
          <strong style={{ color: "#f8fafc" }}>Host:</strong> {hostname} ({hostId})
        </Text>
        <Text style={{ ...baseStyles.text, margin: "0 0 8px" }}>
          <strong style={{ color: "#f8fafc" }}>Metric:</strong> {metric}
        </Text>
        <Text style={{ ...baseStyles.text, margin: "0 0 8px" }}>
          <strong style={{ color: "#f8fafc" }}>Threshold:</strong>{" "}
          <span style={baseStyles.gold}>{threshold}</span>
        </Text>
        <Text style={{ ...baseStyles.text, margin: "0 0 8px" }}>
          <strong style={{ color: "#f8fafc" }}>Actual:</strong>{" "}
          <span style={{ color: "#dc2626", fontWeight: 600 }}>{actual}</span>
        </Text>
        <Text style={{ ...baseStyles.text, margin: 0 }}>
          <strong style={{ color: "#f8fafc" }}>Detected:</strong> {formattedTime}
        </Text>
      </Section>

      <Text style={baseStyles.text}>
        This violation may affect your reputation score and SLA credits. Review
        the host status and take corrective action if needed.
      </Text>

      <Section style={{ textAlign: "center", margin: "32px 0" }}>
        <Link href={`${dashboardUrl}/${hostId}`} style={baseStyles.button}>
          View Host Details
        </Link>
      </Section>

      <Hr style={baseStyles.hr} />

      <Text style={baseStyles.footer}>
        SLA violations are tracked automatically. If this was caused by
        scheduled maintenance, contact{" "}
        <Link href="mailto:support@xcelsior.ca" style={baseStyles.link}>
          support@xcelsior.ca
        </Link>{" "}
        to request an exemption.
      </Text>
    </EmailLayout>
  );
}
