import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ReputationBadge } from "@/components/ui/reputation-badge";

describe("ReputationBadge", () => {
  it("renders the correct label for each tier", () => {
    const tiers = [
      { tier: "new_user", label: "New" },
      { tier: "bronze", label: "Bronze" },
      { tier: "silver", label: "Silver" },
      { tier: "gold", label: "Gold" },
      { tier: "platinum", label: "Platinum" },
      { tier: "diamond", label: "Diamond" },
    ];
    for (const { tier, label } of tiers) {
      const { unmount } = render(<ReputationBadge tier={tier} />);
      expect(screen.getByText(label)).toBeInTheDocument();
      unmount();
    }
  });

  it("falls back to new_user for unknown tier", () => {
    render(<ReputationBadge tier="mythical" />);
    expect(screen.getByText("New")).toBeInTheDocument();
  });

  it("shows score when provided", () => {
    render(<ReputationBadge tier="gold" score={95} />);
    expect(screen.getByText("(95)")).toBeInTheDocument();
  });

  it("does not show score when omitted", () => {
    const { container } = render(<ReputationBadge tier="silver" />);
    expect(container.textContent).not.toContain("(");
  });

  it("handles case-insensitive tier lookup", () => {
    render(<ReputationBadge tier="GOLD" />);
    expect(screen.getByText("Gold")).toBeInTheDocument();
  });
});
