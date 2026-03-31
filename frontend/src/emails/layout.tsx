import {
  Body,
  Container,
  Head,
  Heading,
  Hr,
  Html,
  Img,
  Link,
  Preview,
  Section,
  Text,
} from "@react-email/components";

const baseStyles = {
  body: {
    backgroundColor: "#0a0e1a",
    color: "#f8fafc",
    fontFamily:
      '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  },
  container: {
    maxWidth: "600px",
    margin: "0 auto",
    padding: "40px 24px",
  },
  logo: {
    display: "inline-flex" as const,
    alignItems: "center" as const,
    gap: "12px",
    marginBottom: "32px",
  },
  logoBadge: {
    display: "inline-block",
    width: "36px",
    height: "36px",
    lineHeight: "36px",
    textAlign: "center" as const,
    borderRadius: "8px",
    backgroundColor: "#7c3aed",
    color: "#ffffff",
    fontWeight: 800,
    fontSize: "20px",
  },
  logoText: {
    fontSize: "22px",
    fontWeight: 700,
    color: "#f8fafc",
    textDecoration: "none",
  },
  heading: {
    color: "#f8fafc",
    fontSize: "24px",
    fontWeight: 700,
    lineHeight: 1.3,
    margin: "0 0 16px",
  },
  text: {
    color: "#94a3b8",
    fontSize: "16px",
    lineHeight: 1.6,
    margin: "0 0 16px",
  },
  button: {
    display: "inline-block",
    backgroundColor: "#7c3aed",
    color: "#ffffff",
    padding: "12px 32px",
    borderRadius: "8px",
    fontSize: "16px",
    fontWeight: 600,
    textDecoration: "none",
    textAlign: "center" as const,
  },
  hr: {
    borderColor: "#334155",
    margin: "32px 0",
  },
  footer: {
    color: "#64748b",
    fontSize: "13px",
    lineHeight: 1.5,
  },
  link: {
    color: "#00d4ff",
    textDecoration: "underline",
  },
  card: {
    backgroundColor: "#1e293b",
    borderRadius: "8px",
    padding: "20px 24px",
    border: "1px solid #334155",
    marginBottom: "16px",
  },
  gold: {
    color: "#f59e0b",
    fontWeight: 600,
  },
};

export function EmailLayout({
  preview,
  children,
}: {
  preview: string;
  children: React.ReactNode;
}) {
  return (
    <Html>
      <Head />
      <Preview>{preview}</Preview>
      <Body style={baseStyles.body}>
        <Container style={baseStyles.container}>
          <Section style={baseStyles.logo}>
            <Link href="https://xcelsior.ca" style={{ textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "12px" }}>
              <Img src="https://xcelsior.ca/xcelsior-favicon.svg" width="36" height="36" alt="Xcelsior" style={{ borderRadius: "6px" }} />
              <span style={baseStyles.logoText}>Xcelsior</span>
            </Link>
          </Section>
          {children}
          <Hr style={baseStyles.hr} />
          <Text style={baseStyles.footer}>
            Xcelsior Computing Inc. · Canada
            <br />
            <Link href="https://xcelsior.ca/privacy" style={baseStyles.link}>
              Privacy Policy
            </Link>{" "}
            ·{" "}
            <Link href="https://xcelsior.ca/terms" style={baseStyles.link}>
              Terms
            </Link>
          </Text>
        </Container>
      </Body>
    </Html>
  );
}

export { baseStyles };
