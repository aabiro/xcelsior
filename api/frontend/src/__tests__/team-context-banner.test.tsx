import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { TeamContextBanner } from "@/components/team/team-context-banner";
import type { TeamContext } from "@/lib/team-context";

// The top bar says which workspace is active. Pages repeated it in a
// "{team} workspace" card (owner review, 2026-11-22); only what the workspace
// stops you doing is left.

vi.mock("@/lib/locale", () => ({ useLocale: () => ({ t: (key: string) => key }) }));

function team(overrides: Partial<TeamContext> = {}): TeamContext {
  return {
    isTeamMember: true,
    teamId: "team-1",
    teamName: "aabiro",
    teamRole: "admin",
    billingCustomerId: "team-1",
    canManageBilling: true,
    canWriteInstances: true,
    ...overrides,
  } as TeamContext;
}

describe("TeamContextBanner", () => {
  it.each(["general", "billing", "instances", "volumes", "analytics", "artifacts"] as const)(
    "does not announce the workspace on a %s page",
    (variant) => {
      const { container } = render(<TeamContextBanner team={team()} variant={variant} />);
      expect(container).toBeEmptyDOMElement();
      expect(screen.queryByText("dash.team.banner_title")).toBeNull();
    },
  );

  it("still says when billing is read-only for this member", () => {
    render(<TeamContextBanner team={team({ canManageBilling: false })} variant="billing" />);
    expect(screen.getByText("dash.team.billing_read_only")).toBeInTheDocument();
    expect(screen.queryByText("dash.team.banner_title")).toBeNull();
  });

  it("still says when instances are read-only for a viewer", () => {
    render(<TeamContextBanner team={team({ canWriteInstances: false })} variant="instances" />);
    expect(screen.getByText("dash.team.instances_read_only")).toBeInTheDocument();
  });

  it("still shows the team's instance concurrency", () => {
    render(
      <TeamContextBanner team={team()} variant="instances" concurrency={{ active: 2, cap: 2, shared: true }} />,
    );
    expect(screen.getByText("dash.team.concurrency_usage")).toBeInTheDocument();
  });

  it("renders nothing outside a team", () => {
    const { container } = render(
      <TeamContextBanner team={team({ isTeamMember: false, canManageBilling: false })} variant="billing" />,
    );
    expect(container).toBeEmptyDOMElement();
  });
});
