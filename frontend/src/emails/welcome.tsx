import { Heading, Hr, Link, Section, Text } from "@react-email/components";
import { EmailLayout, baseStyles } from "./layout";

interface WelcomeEmailProps {
  name: string;
  loginUrl?: string;
}

export default function WelcomeEmail({
  name = "there",
  loginUrl = "https://xcelsior.ca/login",
}: WelcomeEmailProps) {
  return (
    <EmailLayout preview={`Welcome to Xcelsior, ${name}!`}>
      <Heading style={baseStyles.heading}>Welcome to Xcelsior</Heading>
      <Text style={baseStyles.text}>
        Hi {name}, thanks for signing up. You now have access to Canada&apos;s
        sovereign GPU compute marketplace.
      </Text>

      <Section style={baseStyles.card}>
        <Text style={{ ...baseStyles.text, margin: "0 0 8px" }}>
          <span style={baseStyles.gold}>✓</span> Full PIPEDA & Law 25
          compliance
        </Text>
        <Text style={{ ...baseStyles.text, margin: "0 0 8px" }}>
          <span style={baseStyles.gold}>✓</span> Data under Canadian law only
        </Text>
        <Text style={{ ...baseStyles.text, margin: 0 }}>
          <span style={baseStyles.gold}>✓</span> Native CAD pricing — no
          conversion fees
        </Text>
      </Section>

      <Text style={baseStyles.text}>
        Get started by submitting your first job or listing your GPUs on the
        marketplace.
      </Text>

      <Section style={{ textAlign: "center", margin: "32px 0" }}>
        <Link href={loginUrl} style={baseStyles.button}>
          Go to Dashboard
        </Link>
      </Section>

      <Hr style={baseStyles.hr} />

      <Text style={baseStyles.text}>
        Questions? Reply to this email or reach us at{" "}
        <Link href="mailto:support@xcelsior.ca" style={baseStyles.link}>
          support@xcelsior.ca
        </Link>
        .
      </Text>
    </EmailLayout>
  );
}
