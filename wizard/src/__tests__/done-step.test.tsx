// Tests for the interactive post-flow deepener menu (Part G).

import React from "react";
import { describe, it, expect, vi, beforeAll } from "vitest";
import { render } from "ink-testing-library";
import { DoneStep, buildDeepenerItems } from "../steps.js";

// Never spawn a real browser from these tests.
beforeAll(() => { process.env.XCELSIOR_NO_BROWSER = "1"; });

const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));
const noop = () => {};

describe("buildDeepenerItems", () => {
    it("provider gets the worker-install item, no instance item", () => {
        const items = buildDeepenerItems("provide", "https://x");
        const labels = items.map((i) => i.label).join("\n");
        expect(labels).toContain("xcelsior worker install");
        expect(labels).not.toContain("Manage instances");
    });

    it("renter gets the instance manager item, no worker item", () => {
        const items = buildDeepenerItems("rent", "https://x");
        const labels = items.map((i) => i.label).join("\n");
        expect(labels).toContain("Manage instances");
        expect(labels).not.toContain("xcelsior worker install");
    });

    it("both gets worker + instance items", () => {
        const labels = buildDeepenerItems("both", "https://x").map((i) => i.label).join("\n");
        expect(labels).toContain("xcelsior worker install");
        expect(labels).toContain("Manage instances");
    });

    it("always ends with a Finish & exit item, and includes notifications + dashboard", () => {
        for (const mode of ["rent", "provide", "both"]) {
            const items = buildDeepenerItems(mode, "https://x");
            expect(items[items.length - 1].value).toBe("exit");
            const vals = items.map((i) => i.value).join(" ");
            expect(vals).toContain("url:https://x/dashboard/settings/notifications");
            expect(vals).toContain("url:https://x/dashboard");
        }
    });
});

describe("DoneStep", () => {
    it("renders the Go further menu for providers", () => {
        const { lastFrame } = render(
            <DoneStep answers={{ mode: "provide", _api_base_url: "https://xcelsior.ca" }} onExit={noop} />,
        );
        const out = lastFrame() ?? "";
        expect(out).toContain("Go further");
        expect(out).toContain("xcelsior worker install");
        expect(out).toContain("notifications");
        expect(out).toContain("Finish & exit");
    });

    it("shows instance details when present", () => {
        const { lastFrame } = render(
            <DoneStep
                answers={{ mode: "rent", _api_base_url: "https://xcelsior.ca" }}
                instanceInfo={{ job_id: "job-123", host_ip: "10.0.0.1", ssh_port: 2222, status: "starting" }}
                onExit={noop}
            />,
        );
        const out = lastFrame() ?? "";
        expect(out).toContain("Instance launched!");
        expect(out).toContain("job-123");
        expect(out).toContain("ssh -p 2222");
    });

    it("selecting the first item shows a note instead of exiting (provider → worker note)", async () => {
        const onExit = vi.fn();
        const { lastFrame, stdin } = render(
            <DoneStep answers={{ mode: "provide", _api_base_url: "https://xcelsior.ca" }} onExit={onExit} />,
        );
        await delay(10);
        stdin.write("\r"); // select the highlighted first item (the worker note)
        await delay(30);
        expect(onExit).not.toHaveBeenCalled();
        expect(lastFrame()).toContain("credentials are already saved");
    });

    it("pressing q exits immediately", async () => {
        const onExit = vi.fn();
        const { stdin } = render(
            <DoneStep answers={{ mode: "rent", _api_base_url: "https://xcelsior.ca" }} onExit={onExit} />,
        );
        await delay(10);
        stdin.write("q");
        await delay(20);
        expect(onExit).toHaveBeenCalledTimes(1);
    });
});
