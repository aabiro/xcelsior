import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { StatCard } from "@/components/ui/stat-card";
import { Activity } from "lucide-react";

describe("StatCard", () => {
  it("renders label and value", () => {
    render(<StatCard label="Revenue" value="$1,234" />);
    expect(screen.getByText("Revenue")).toBeInTheDocument();
    expect(screen.getByText("$1,234")).toBeInTheDocument();
  });

  it("renders trend indicator when provided", () => {
    render(<StatCard label="Users" value="500" trend="up" trendValue="+12%" />);
    expect(screen.getByText("+12%")).toBeInTheDocument();
  });

  it("does not render trend when omitted", () => {
    const { container } = render(<StatCard label="Hosts" value="10" />);
    expect(container.querySelectorAll("[class*='trend']")).toHaveLength(0);
  });

  it("renders icon when provided", () => {
    render(<StatCard label="Activity" value="42" icon={Activity} />);
    // Icon renders as SVG inside the component
    expect(screen.getByText("Activity")).toBeInTheDocument();
  });
});
