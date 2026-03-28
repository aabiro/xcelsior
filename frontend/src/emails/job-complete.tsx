import { Heading, Hr, Link, Section, Text } from "@react-email/components";
import { EmailLayout, baseStyles } from "./layout";

interface JobCompleteProps {
  name: string;
  jobId: string;
  jobName: string;
  duration: string;
  cost: string;
  dashboardUrl?: string;
}

export default function JobCompleteEmail({
  name = "there",
  jobId = "job-8f4a2c",
  jobName = "llama-finetune-v3",
  duration = "14h 23m",
  cost = "$108.42 CAD",
  dashboardUrl = "https://xcelsior.ca/dashboard/jobs",
}: JobCompleteProps) {
  return (
    <EmailLayout preview={`Job ${jobName} completed`}>
      <Heading style={baseStyles.heading}>Job Completed</Heading>
      <Text style={baseStyles.text}>
        Hi {name}, your job has finished successfully.
      </Text>

      <Section style={baseStyles.card}>
        <Text style={{ ...baseStyles.text, margin: "0 0 8px" }}>
          <strong style={{ color: "#f8fafc" }}>Job:</strong> {jobName}
        </Text>
        <Text style={{ ...baseStyles.text, margin: "0 0 8px" }}>
          <strong style={{ color: "#f8fafc" }}>ID:</strong> {jobId}
        </Text>
        <Text style={{ ...baseStyles.text, margin: "0 0 8px" }}>
          <strong style={{ color: "#f8fafc" }}>Duration:</strong> {duration}
        </Text>
        <Text style={{ ...baseStyles.text, margin: 0 }}>
          <strong style={{ color: "#f8fafc" }}>Cost:</strong>{" "}
          <span style={baseStyles.gold}>{cost}</span>
        </Text>
      </Section>

      <Section style={{ textAlign: "center", margin: "32px 0" }}>
        <Link href={`${dashboardUrl}/${jobId}`} style={baseStyles.button}>
          View Results
        </Link>
      </Section>

      <Hr style={baseStyles.hr} />

      <Text style={baseStyles.text}>
        Artifacts are available for download from the job detail page. They will
        be retained for 30 days.
      </Text>
    </EmailLayout>
  );
}
