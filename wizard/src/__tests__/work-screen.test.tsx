// Render + behavior tests for the Phase 2/3 panes: TaskListPane, LearnSlides,
// WorkScreen. Covers slideshow advance/teardown and reflow threshold.

import React, { act } from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render } from "ink-testing-library";
import { TaskListPane } from "../TaskListPane.js";
import { LearnSlides } from "../LearnSlides.js";
import { WorkScreen, shouldReflow, REFLOW_COLUMNS, cycleTab } from "../WorkScreen.js";
import { buildTaskList, computeTaskStates } from "../task-model.js";

const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));

afterEach(() => {
    vi.useRealTimers();
});

const sampleTasks = computeTaskStates(buildTaskList("provide"), {
    currentStepId: "benchmark",
    completedStepIds: ["mode", "docker-check", "device-auth", "api-check", "gpu-detect", "version-check", "network-setup"],
});

describe("TaskListPane", () => {
    it("renders task labels, glyphs and progress", () => {
        const { lastFrame } = render(<TaskListPane tasks={sampleTasks} />);
        const out = lastFrame() ?? "";
        expect(out).toContain("Tasks");
        expect(out).toContain("Benchmark GPU compute");
        expect(out).toContain("Progress:");
        expect(out).toContain("■"); // a done glyph
        expect(out).toContain("▶"); // active glyph
    });
});

describe("LearnSlides", () => {
    it("renders the first slide heading", () => {
        const { lastFrame } = render(<LearnSlides intervalMs={1000} />);
        expect(lastFrame()).toContain("Learn");
    });

    it("cycles multi-slide decks and tears the interval down on unmount", async () => {
        const slides = [
            { mode: "learn" as const, heading: "First Slide AAA", lines: ["a"] },
            { mode: "learn" as const, heading: "Second Slide BBB", lines: ["b"] },
        ];
        const { lastFrame, unmount } = render(<LearnSlides slides={slides} intervalMs={40} />);
        expect(lastFrame()).toContain("First Slide AAA");
        // ink-testing-library does not always flush setInterval ticks; poll briefly.
        await act(async () => {
            await vi.waitFor(
                () => expect(lastFrame()).toContain("Second Slide BBB"),
                { timeout: 2000, interval: 20 },
            );
        });
        await act(async () => {
            await vi.waitFor(
                () => expect(lastFrame()).toContain("First Slide AAA"),
                { timeout: 2000, interval: 20 },
            );
        });
        unmount();
        const frozen = lastFrame();
        await act(async () => {
            await delay(120);
        });
        expect(lastFrame()).toBe(frozen);
    });

    it("renders a step-aware intro and an interactive nudge", () => {
        const { lastFrame } = render(
            <LearnSlides intervalMs={100000}
                intro="Benchmarks take ~60s — XCU scoring while you wait"
                nudge="Press A to ask Hexara anything" />,
        );
        const out = lastFrame() ?? "";
        expect(out).toContain("Benchmarks take ~60s");
        expect(out).toContain("Press A to ask Hexara");
    });

    it("renders a single slide without advancing", async () => {
        vi.useFakeTimers();
        const slides = [{ mode: "learn" as const, heading: "Solo Slide", lines: ["x"] }];
        const { lastFrame, unmount } = render(<LearnSlides slides={slides} intervalMs={1000} />);
        expect(lastFrame()).toContain("Solo Slide");
        await vi.advanceTimersByTimeAsync(5000);
        expect(lastFrame()).toContain("Solo Slide"); // nothing to advance to
        unmount();
    });
});

describe("WorkScreen", () => {
    it("renders both panes", () => {
        const { lastFrame } = render(<WorkScreen tasks={sampleTasks} slideIntervalMs={100000} />);
        const out = lastFrame() ?? "";
        expect(out).toContain("Tasks");
        expect(out).toContain("Learn");
    });

    it("renders the status strip with the most recent log lines", () => {
        const { lastFrame } = render(
            <WorkScreen tasks={sampleTasks} slideIntervalMs={100000}
                log={["old line", "Running GPU benchmarks…", "Hexara is analyzing 2 issues…"]} />,
        );
        const out = lastFrame() ?? "";
        expect(out).toContain("Status");
        expect(out).toContain("Hexara is analyzing 2 issues…");
    });

    it("shouldReflow flips below the threshold", () => {
        expect(shouldReflow(REFLOW_COLUMNS - 1)).toBe(true);
        expect(shouldReflow(REFLOW_COLUMNS)).toBe(false);
        expect(shouldReflow(140)).toBe(false);
        expect(shouldReflow(60)).toBe(true);
    });

    it("shows the tab bar and an Ask Hexara hint", () => {
        const { lastFrame } = render(
            <WorkScreen tasks={sampleTasks} slideIntervalMs={100000}
                log={["running…"]} tail={["✓ runc: v1.2.1"]} />,
        );
        const out = lastFrame() ?? "";
        expect(out).toContain("Status");
        expect(out).toContain("Tail logs");
        expect(out).toContain("Ask Hexara");
        // default tab is Status → tail content not shown yet
        expect(out).not.toContain("✓ runc: v1.2.1");
    });

    it("reaches the Ask Hexara tab after two Tab presses", async () => {
        const { lastFrame, stdin } = render(
            <WorkScreen tasks={sampleTasks} slideIntervalMs={100000} log={["x"]} tail={["y"]} />,
        );
        await delay(10);
        stdin.write("\t"); // → tail
        await delay(20);
        stdin.write("\t"); // → ask
        await delay(20);
        expect(lastFrame()).toContain("ask Hexara about this step");
    });

    it("switches to the Tail logs tab on Tab keypress", async () => {
        const { lastFrame, stdin } = render(
            <WorkScreen tasks={sampleTasks} slideIntervalMs={100000}
                log={["running…"]} tail={["✓ runc: v1.2.1", "✗ docker: not found"]} />,
        );
        await delay(10); // let useInput mount
        expect(lastFrame()).not.toContain("runc: v1.2.1");
        stdin.write("\t"); // Tab → next tab
        await delay(50);
        const out = lastFrame() ?? "";
        expect(out).toContain("✓ runc: v1.2.1");
        expect(out).toContain("✗ docker: not found");
    });
});

describe("cycleTab", () => {
    it("cycles forward through all three tabs and wraps", () => {
        expect(cycleTab("status", 1)).toBe("tail");
        expect(cycleTab("tail", 1)).toBe("ask");
        expect(cycleTab("ask", 1)).toBe("status");
    });
    it("cycles backward and wraps", () => {
        expect(cycleTab("status", -1)).toBe("ask");
        expect(cycleTab("ask", -1)).toBe("tail");
        expect(cycleTab("tail", -1)).toBe("status");
    });
});
