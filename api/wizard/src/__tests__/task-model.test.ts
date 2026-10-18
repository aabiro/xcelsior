// Tests for task-model.ts — task derivation and state projection.

import { describe, it, expect } from "vitest";
import {
    buildTaskList, computeTaskStates, taskProgress, labelForStep,
} from "../task-model.js";
import { WIZARD_STEPS } from "../wizard-flow.js";

describe("buildTaskList", () => {
    it("returns empty list when mode is undefined", () => {
        expect(buildTaskList(undefined)).toEqual([]);
    });

    it("renter list excludes provider-only steps and the done step", () => {
        const tasks = buildTaskList("rent");
        const ids = tasks.map((t) => t.id);
        expect(ids).toContain("mode");
        expect(ids).toContain("device-auth");
        expect(ids).toContain("launch-instance");
        expect(ids).not.toContain("benchmark"); // provider-only
        expect(ids).not.toContain("host-register");
        expect(ids).not.toContain("done");
    });

    it("provider list includes provider steps and excludes renter-only steps", () => {
        const tasks = buildTaskList("provide");
        const ids = tasks.map((t) => t.id);
        expect(ids).toContain("benchmark");
        expect(ids).toContain("verification");
        expect(ids).toContain("host-register");
        expect(ids).not.toContain("browse-gpus"); // renter-only
        expect(ids).not.toContain("launch-instance");
    });

    it("excludes optional runtime-gated steps from the stable denominator", () => {
        const ids = buildTaskList("provide").map((t) => t.id);
        expect(ids).not.toContain("custom-rate"); // only if pricing === custom
        expect(ids).not.toContain("spot-min-cents");
    });

    it("every task gets a human label, never the raw id", () => {
        for (const task of buildTaskList("both")) {
            expect(task.label.length).toBeGreaterThan(0);
            // label should differ from id for known steps
            const step = WIZARD_STEPS.find((s) => s.id === task.id)!;
            expect(task.label).toBe(labelForStep(step));
        }
    });

    it("all tasks start as todo", () => {
        expect(buildTaskList("rent").every((t) => t.state === "todo")).toBe(true);
    });
});

describe("computeTaskStates", () => {
    const tasks = buildTaskList("rent");

    it("marks completed steps as done", () => {
        const out = computeTaskStates(tasks, {
            currentStepId: "device-auth",
            completedStepIds: ["mode"],
        });
        expect(out.find((t) => t.id === "mode")!.state).toBe("done");
    });

    it("marks the current step as active", () => {
        const out = computeTaskStates(tasks, {
            currentStepId: "device-auth",
            completedStepIds: ["mode"],
        });
        expect(out.find((t) => t.id === "device-auth")!.state).toBe("active");
    });

    it("marks the current step failed when its check failed", () => {
        const out = computeTaskStates(tasks, {
            currentStepId: "api-check",
            completedStepIds: ["mode", "device-auth"],
            currentFailed: true,
        });
        expect(out.find((t) => t.id === "api-check")!.state).toBe("failed");
    });

    it("derives done from passing check results even if not in completed list", () => {
        const out = computeTaskStates(tasks, {
            currentStepId: "gpu-pick",
            completedStepIds: [],
            checkResults: { "api-check": { items: [{ name: "x", ok: true, detail: "" }], allPassed: true } },
        });
        expect(out.find((t) => t.id === "api-check")!.state).toBe("done");
    });

    it("derives failed from failing check results", () => {
        const out = computeTaskStates(tasks, {
            currentStepId: "gpu-pick",
            completedStepIds: [],
            checkResults: { "api-check": { items: [{ name: "x", ok: false, detail: "" }], allPassed: false } },
        });
        expect(out.find((t) => t.id === "api-check")!.state).toBe("failed");
    });

    it("supports concurrent active tasks via alsoActive", () => {
        const ptasks = buildTaskList("provide");
        const out = computeTaskStates(ptasks, {
            currentStepId: "network-setup",
            completedStepIds: [],
            alsoActive: ["host-register"],
        });
        expect(out.find((t) => t.id === "network-setup")!.state).toBe("active");
        expect(out.find((t) => t.id === "host-register")!.state).toBe("active");
    });

    it("leaves untouched future steps as todo", () => {
        const out = computeTaskStates(tasks, {
            currentStepId: "mode",
            completedStepIds: [],
        });
        expect(out.find((t) => t.id === "launch-instance")!.state).toBe("todo");
    });
});

describe("taskProgress", () => {
    it("counts done over total", () => {
        const tasks = buildTaskList("rent");
        const projected = computeTaskStates(tasks, {
            currentStepId: "device-auth",
            completedStepIds: ["mode"],
        });
        const p = taskProgress(projected);
        expect(p.done).toBe(1);
        expect(p.total).toBe(tasks.length);
    });
});
