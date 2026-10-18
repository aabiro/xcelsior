import React from "react";
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";

/**
 * Gate P4's frontend clause: *"a pipeline view showing the graph, which stage
 * is live, and one approval covering all of it — with the total committed spend
 * stated **before** approval, not after."*
 *
 * Three properties are worth asserting and one is worth more than the others.
 *
 * The ceiling must be visible **before** approval, when nothing has been spent.
 * That is the whole point of the clause: a total shown only once the run is
 * underway is a report, not a decision the user got to make.
 *
 * `on_failure` must be legible. `docs/pipeline-plan.md` §3.2 makes it part of
 * what is approved, and it previously reached the page inside a serialised
 * object in one table cell — present, and not readable by anyone.
 *
 * And the state vocabulary must survive a value this component does not know.
 * The five states come from `ck_pipeline_stage_state`; a sixth added later must
 * render as *something*, because a blank cell is worse than an ugly label.
 */

import { PipelineGraph } from "@/components/pipelines/pipeline-graph";

const GRAPH = [
  { index: 0, name: "train", action_type: "create_instance", on_failure: "halt" as const, max_attempts: 1, estimate_micros: 4_000_000 },
  { index: 1, name: "evaluate", action_type: "create_instance", on_failure: "retry" as const, max_attempts: 3, estimate_micros: 1_000_000 },
  { index: 2, name: "serve", action_type: "deploy_endpoint", on_failure: "continue" as const, max_attempts: 1, estimate_micros: 2_000_000 },
];

describe("before approval", () => {
  it("states the total the user is agreeing to, with nothing spent yet", () => {
    render(<PipelineGraph stages={GRAPH} approvedMaxMicros={7_000_000} />);
    expect(screen.getByTestId("pipeline-ceiling")).toHaveTextContent("$7.00");
    // The clause is specifically "before approval, not after" — a spend figure
    // here would imply the run already started.
    expect(screen.queryByTestId("pipeline-spent")).toBeNull();
    expect(screen.queryByTestId("pipeline-spend-bar")).toBeNull();
  });

  it("calls it a ceiling rather than an estimate", () => {
    render(<PipelineGraph stages={GRAPH} approvedMaxMicros={7_000_000} />);
    expect(screen.getByText(/ceiling, not an estimate/i)).toBeInTheDocument();
  });

  it("shows every stage in the declared order", () => {
    render(<PipelineGraph stages={GRAPH} approvedMaxMicros={7_000_000} />);
    const names = screen.getAllByTestId(/^pipeline-stage-\d+$/).map((el) => el.textContent ?? "");
    expect(names[0]).toContain("train");
    expect(names[1]).toContain("evaluate");
    expect(names[2]).toContain("serve");
  });

  it("makes each stage's failure behaviour readable, not serialised", () => {
    render(<PipelineGraph stages={GRAPH} approvedMaxMicros={7_000_000} />);
    expect(screen.getByTestId("pipeline-stage-0-failure")).toHaveTextContent(
      /pipeline stops here/i,
    );
    // The bound matters: unbounded retry inside a ceiling is a way to spend the
    // whole ceiling on a stage that cannot work (§3.2).
    expect(screen.getByTestId("pipeline-stage-1-failure")).toHaveTextContent(/up to 3/i);
    expect(screen.getByTestId("pipeline-stage-2-failure")).toHaveTextContent(/carries on/i);
  });

  it("defaults every stage to waiting when no run exists", () => {
    render(<PipelineGraph stages={GRAPH} approvedMaxMicros={7_000_000} />);
    for (const i of [0, 1, 2]) {
      expect(screen.getByTestId(`pipeline-stage-${i}`)).toHaveAttribute("data-state", "pending");
    }
  });
});

describe("while running", () => {
  const RUNNING = [
    { ...GRAPH[0], state: "succeeded", attempt_count: 1, spent_micros: 3_500_000 },
    { ...GRAPH[1], state: "running", attempt_count: 2, spent_micros: 0 },
    { ...GRAPH[2], state: "pending", attempt_count: 0, spent_micros: 0 },
  ];

  it("marks which stage is live", () => {
    render(<PipelineGraph stages={RUNNING} approvedMaxMicros={7_000_000} spentMicros={3_500_000} approved />);
    expect(screen.getByTestId("pipeline-stage-1")).toHaveAttribute("data-state", "running");
    expect(screen.getByTestId("pipeline-stage-1-state")).toHaveTextContent(/running/i);
    expect(screen.getByTestId("pipeline-stage-0-state")).toHaveTextContent(/done/i);
  });

  it("shows the retry attempt, so a silent loop is visible", () => {
    render(<PipelineGraph stages={RUNNING} approvedMaxMicros={7_000_000} spentMicros={3_500_000} approved />);
    expect(screen.getByTestId("pipeline-stage-1-state")).toHaveTextContent(/attempt 2/i);
  });

  it("shows spend against the ceiling, not against a forecast", () => {
    render(<PipelineGraph stages={RUNNING} approvedMaxMicros={7_000_000} spentMicros={3_500_000} approved />);
    expect(screen.getByTestId("pipeline-spent")).toHaveTextContent("$3.50");
    expect(screen.getByTestId("pipeline-ceiling")).toHaveTextContent("$7.00");
    expect(screen.getByTestId("pipeline-spend-bar")).toHaveAttribute("aria-valuenow", "50");
  });

  it("never draws the bar past full, however the numbers arrive", () => {
    // The executor stops a stage that would exceed the ceiling, so this should
    // not occur — but a bar rendered at 300% would be a visual claim that the
    // guarantee failed, which is a worse way to find out than a number.
    render(<PipelineGraph stages={RUNNING} approvedMaxMicros={1_000_000} spentMicros={3_500_000} approved />);
    expect(screen.getByTestId("pipeline-spend-bar")).toHaveAttribute("aria-valuenow", "100");
  });

  it("surfaces a failure code rather than only a red icon", () => {
    const failed = [{ ...GRAPH[0], state: "failed", attempt_count: 1, spent_micros: 0, failure_code: "budget_exceeded" }];
    render(<PipelineGraph stages={failed} approvedMaxMicros={7_000_000} spentMicros={0} approved />);
    expect(screen.getByText("budget_exceeded")).toBeInTheDocument();
  });
});

describe("robustness", () => {
  it("renders an unknown stage state instead of a blank cell", () => {
    const odd = [{ ...GRAPH[0], state: "quarantined", attempt_count: 0, spent_micros: 0 }];
    render(<PipelineGraph stages={odd} approvedMaxMicros={1_000_000} spentMicros={0} approved />);
    expect(screen.getByTestId("pipeline-stage-0-state")).toHaveTextContent("quarantined");
  });

  it("does not divide by a zero ceiling", () => {
    render(<PipelineGraph stages={GRAPH} approvedMaxMicros={0} spentMicros={0} approved />);
    expect(screen.getByTestId("pipeline-spend-bar")).toHaveAttribute("aria-valuenow", "0");
  });
});
