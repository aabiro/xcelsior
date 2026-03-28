import { Heading, Hr, Link, Section, Text } from "@react-email/components";
import { EmailLayout, baseStyles } from "./layout";

interface InvoiceEmailProps {
  name: string;
  invoiceNumber: string;
  amount: string;
  period: string;
  dashboardUrl?: string;
  items: { description: string; hours: string; total: string }[];
}

export default function InvoiceEmail({
  name = "there",
  invoiceNumber = "INV-2026-0042",
  amount = "$1,247.80 CAD",
  period = "February 2026",
  dashboardUrl = "https://xcelsior.ca/dashboard/billing",
  items = [
    { description: "4x A100 80GB — Training job #1842", hours: "142h", total: "$1,073.52 CAD" },
    { description: "2x RTX 4090 — Inference batch #921", hours: "38h", total: "$174.28 CAD" },
  ],
}: InvoiceEmailProps) {
  return (
    <EmailLayout preview={`Xcelsior Invoice ${invoiceNumber} — ${amount}`}>
      <Heading style={baseStyles.heading}>Invoice {invoiceNumber}</Heading>
      <Text style={baseStyles.text}>
        Hi {name}, here&apos;s your invoice for {period}.
      </Text>

      <Section style={baseStyles.card}>
        {items.map((item, i) => (
          <div key={i}>
            <Text style={{ ...baseStyles.text, margin: "0 0 4px", fontWeight: 600, color: "#f8fafc" }}>
              {item.description}
            </Text>
            <Text style={{ ...baseStyles.text, margin: i < items.length - 1 ? "0 0 16px" : "0" }}>
              {item.hours} — <span style={baseStyles.gold}>{item.total}</span>
            </Text>
          </div>
        ))}
      </Section>

      <Section
        style={{
          ...baseStyles.card,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <Text style={{ ...baseStyles.text, margin: 0 }}>Total</Text>
        <Text
          style={{
            color: "#f8fafc",
            fontSize: "24px",
            fontWeight: 800,
            margin: 0,
          }}
        >
          {amount}
        </Text>
      </Section>

      <Section style={{ textAlign: "center", margin: "32px 0" }}>
        <Link href={dashboardUrl} style={baseStyles.button}>
          View in Dashboard
        </Link>
      </Section>

      <Hr style={baseStyles.hr} />

      <Text style={baseStyles.footer}>
        Payment is processed automatically via your card on file. If you have
        questions, reply to this email.
      </Text>
    </EmailLayout>
  );
}
