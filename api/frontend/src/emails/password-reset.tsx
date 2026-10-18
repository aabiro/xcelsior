import { Heading, Link, Section, Text } from "@react-email/components";
import { EmailLayout, baseStyles } from "./layout";

interface PasswordResetProps {
  name: string;
  resetUrl: string;
}

export default function PasswordResetEmail({
  name = "there",
  resetUrl = "https://xcelsior.ca/reset-password?token=example",
}: PasswordResetProps) {
  return (
    <EmailLayout preview="Reset your Xcelsior password">
      <Heading style={baseStyles.heading}>Password Reset</Heading>
      <Text style={baseStyles.text}>
        Hi {name}, we received a request to reset your password. Click the
        button below to choose a new one.
      </Text>

      <Section style={{ textAlign: "center", margin: "32px 0" }}>
        <Link href={resetUrl} style={baseStyles.button}>
          Reset Password
        </Link>
      </Section>

      <Text style={baseStyles.text}>
        This link expires in 1 hour. If you didn&apos;t request a reset, you can
        safely ignore this email.
      </Text>

      <Text style={{ ...baseStyles.footer, marginTop: "24px" }}>
        If the button doesn&apos;t work, copy and paste this URL into your
        browser:
        <br />
        <Link href={resetUrl} style={baseStyles.link}>
          {resetUrl}
        </Link>
      </Text>
    </EmailLayout>
  );
}
