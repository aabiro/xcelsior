import React from "react";
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";

/**
 * Gate P6: *"Provider dashboard parity: the same earnings and payout state the
 * tools return."*
 *
 * Earnings already matched field for field. Payout state did not — the route
 * returns a `payouts` object, `get_provider_account` hands the whole thing to
 * the model, and the dashboard's locally re-declared provider type stopped at
 * `paypal`. So a provider asking their agent got a list of what Stripe needs,
 * and the same provider on the dashboard got the word "Restricted".
 *
 * The property that matters most here is not "does it list requirements" — it
 * is that **absent information never renders as good news**. This is the one
 * surface whose entire job is explaining why someone is not being paid, and
 * `checked_live: false` means every other field is a zero value rather than a
 * measurement.
 */

import {
  PayoutRequirements,
  readableRequirement,
  hasExplanation,
} from "@/components/providers/payout-requirements";

const READY = {
  charges_enabled: true,
  payouts_enabled: true,
  currently_due: [],
  past_due: [],
  disabled_reason: "",
  checked_live: true,
};

describe("when Stripe was never actually consulted", () => {
  it("says so rather than showing an empty requirement list as success", () => {
    // Every field is its zero value. A naive renderer reads this as
    // "payouts_enabled: false, nothing due" and says "you're all set".
    render(
      <PayoutRequirements
        payouts={{ ...READY, payouts_enabled: false, charges_enabled: false, checked_live: false }}
      />,
    );
    expect(screen.getByTestId("payout-state-unknown")).toBeInTheDocument();
    expect(screen.queryByTestId("payout-state-enabled")).toBeNull();
  });

  it("does not claim everything is in order", () => {
    render(<PayoutRequirements payouts={{ ...READY, checked_live: false }} />);
    // checked_live false with payouts_enabled TRUE is the nastiest case: the
    // stale flag says yes and nothing checked it.
    expect(screen.getByTestId("payout-state-unknown")).toBeInTheDocument();
    expect(screen.getByText(/not a statement that everything is in order/i)).toBeInTheDocument();
  });

  it("treats a missing object the same as an unchecked one", () => {
    render(<PayoutRequirements payouts={undefined} />);
    expect(screen.getByTestId("payout-state-unknown")).toBeInTheDocument();
  });
});

describe("when Stripe answered", () => {
  it("confirms payouts only when nothing is outstanding", () => {
    render(<PayoutRequirements payouts={READY} />);
    expect(screen.getByTestId("payout-state-enabled")).toBeInTheDocument();
  });

  it("does not call it ready when a requirement is pending, even if enabled", () => {
    // Stripe leaves payouts on while a future-deadline requirement accrues.
    // Telling the provider early is the whole value of showing this.
    render(
      <PayoutRequirements payouts={{ ...READY, currently_due: ["individual.id_number"] }} />,
    );
    expect(screen.queryByTestId("payout-state-enabled")).toBeNull();
    expect(screen.getByTestId("payout-state-blocked")).toBeInTheDocument();
  });

  it("lists what is needed, so the provider knows what to do", () => {
    render(
      <PayoutRequirements
        payouts={{ ...READY, payouts_enabled: false, currently_due: ["external_account"] }}
      />,
    );
    expect(screen.getByTestId("requirements-currently-due")).toHaveTextContent(
      /bank account/i,
    );
  });

  it("separates overdue from merely needed", () => {
    render(
      <PayoutRequirements
        payouts={{
          ...READY,
          payouts_enabled: false,
          past_due: ["external_account"],
          currently_due: ["business_profile.url"],
        }}
      />,
    );
    expect(screen.getByTestId("requirements-past-due")).toHaveTextContent(/bank account/i);
    expect(screen.getByTestId("requirements-currently-due")).toHaveTextContent(/website/i);
  });

  it("shows the disabled reason when Stripe gave one", () => {
    render(
      <PayoutRequirements
        payouts={{ ...READY, payouts_enabled: false, disabled_reason: "requirements.past_due" }}
      />,
    );
    expect(screen.getByTestId("payout-disabled-reason")).toHaveTextContent("requirements.past_due");
  });
});

describe("the glossary annotates rather than gates", () => {
  it("explains the requirements that actually occur", () => {
    expect(readableRequirement("external_account")).toMatch(/bank account/i);
    expect(readableRequirement("individual.verification.document")).toMatch(/photo/i);
  });

  it("passes an unknown requirement through unchanged, not re-spaced", () => {
    // A mechanical transform would return "Some future requirement" here, which
    // reads like a translation and carries no more information than the token.
    // Leaving it raw is what lets a reader tell explained from unexplained.
    expect(readableRequirement("some_future.requirement")).toBe("some_future.requirement");
    expect(hasExplanation("some_future.requirement")).toBe(false);
  });

  it("still renders and still blocks on a requirement it cannot explain", () => {
    // The `min_tier` failure was a hand-maintained map that *rejected* unlisted
    // values. This one must never skip or swallow an unknown requirement.
    render(
      <PayoutRequirements
        payouts={{ ...READY, payouts_enabled: false, currently_due: ["some_future.requirement"] }}
      />,
    );
    expect(screen.getByTestId("payout-state-blocked")).toBeInTheDocument();
    expect(screen.getByTestId("requirements-currently-due")).toHaveTextContent(
      "some_future.requirement",
    );
  });

  it("has no entry that merely restates its own key", () => {
    // The guard against the glossary drifting back into decoration: an entry
    // that is just its key with different spacing is not an explanation.
    const keys = [
      "external_account",
      "individual.id_number",
      "individual.verification.document",
      "business_profile.url",
      "company.tax_id",
      "tos_acceptance.date",
    ];
    for (const key of keys) {
      const label = readableRequirement(key);
      const flattened = key.replaceAll(".", " ").replaceAll("_", " ").toLowerCase();
      expect(label.toLowerCase()).not.toBe(flattened);
    }
  });
});
