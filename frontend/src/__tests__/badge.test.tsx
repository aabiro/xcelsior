import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { Badge, StatusBadge } from "@/components/ui/badge";

describe("Badge", () => {
  it("renders children", () => {
    render(<Badge>Active</Badge>);
    expect(screen.getByText("Active")).toBeInTheDocument();
  });

  it("applies variant classes", () => {
    const { container } = render(<Badge variant="active">Online</Badge>);
    expect(container.firstChild).toHaveClass("text-emerald");
  });
});

describe("StatusBadge", () => {
  it.each([
    ["active", "Active"],
    ["dead", "Dead"],
    ["draining", "Draining"],
    ["queued", "Queued"],
    ["running", "Running"],
    ["completed", "Completed"],
    ["failed", "Failed"],
    ["cancelled", "Cancelled"],
  ])("renders status '%s'", (status, label) => {
    render(<StatusBadge status={status} />);
    expect(screen.getByText(label)).toBeInTheDocument();
  });

  it("falls back to default for unknown status", () => {
    render(<StatusBadge status="unknown_status" />);
    expect(screen.getByText("Unknown_status")).toBeInTheDocument();
  });
});
