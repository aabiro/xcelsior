import { Heading, Link, Section, Text } from "@react-email/components";
import { EmailLayout, baseStyles } from "./layout";

interface LowCreditsProps {
  name: string;
  balance: string;
  topUpUrl?: string;
}

export default function LowCreditsEmail({
  name = "there",
  balance = "$12.40 CAD",
  topUpUrl = "https://xcelsior.ca/dashboard/billing",
}: LowCreditsProps) {
  return (
    <EmailLayout preview={`Low balance: ${balance} remaining`}>
      <Heading style={baseStyles.heading}>Low Credit Balance</Heading>
      <Text style={baseStyles.text}>
        Hi {name}, your Xcelsior balance is{" "}
        <span style={baseStyles.gold}>{balance}</span>. Running jobs may be
        paused if your balance reaches zero.
      </Text>

      <Section style={{ textAlign: "center", margin: "32px 0" }}>
        <Link href={topUpUrl} style={baseStyles.button}>
          Add Credits
        </Link>
      </Section>

      <Text style={baseStyles.text}>
        To avoid interruptions, we recommend keeping at least 24 hours of
        compute credits in your account.
      </Text>
    </EmailLayout>
  );
}
