// useWizardFlow — Hook that drives the structured wizard step-by-step.
// Manages step progression, answer collection, auto-checks, and AI escape hatch.

import { useState, useCallback, useRef } from "react";
import { WIZARD_STEPS, getNextStep, type WizardStep } from "./wizard-flow.js";
import { streamChat, type ApiClientConfig, type SSEEvent } from "./api-client.js";
import type { WizardState } from "./wizard-sprite.js";
import { checkDocker, type CheckResult } from "./checks.js";

export interface AutoCheckResults {
    items: CheckResult[];
    allPassed: boolean;
}

export interface UseWizardFlowReturn {
    /** Current step definition */
    step: WizardStep;
    /** Index into WIZARD_STEPS */
    stepIndex: number;
    /** All collected answers keyed by step ID */
    answers: Record<string, string | string[]>;
    /** Wizard sprite state */
    wizardState: WizardState;
    /** Wizard sprite message */
    wizardMessage: string;
    /** Results from auto-check steps */
    checkResults: Record<string, AutoCheckResults>;
    /** AI escape hatch response (shown inline, then cleared) */
    aiResponse: string | null;
    /** Whether AI is currently streaming */
    aiStreaming: boolean;
    /** Submit an answer for the current step */
    submitAnswer: (value: string | string[]) => void;
    /** Send free-form text to AI (escape hatch) */
    askAi: (question: string) => Promise<void>;
    /** Dismiss AI response and return to current step */
    dismissAi: () => void;
    /** Whether the wizard flow is complete */
    isComplete: boolean;
}

/** Validate an API connection by hitting /healthz */
async function checkApi(baseUrl: string): Promise<CheckResult[]> {
    try {
        const url = new URL("/healthz", baseUrl || "https://xcelsior.ca");
        const resp = await fetch(url.toString(), { signal: AbortSignal.timeout(10_000) });
        if (resp.ok) {
            return [{ name: "API Connection", ok: true, detail: `${url.origin} — healthy` }];
        }
        return [{ name: "API Connection", ok: false, detail: `HTTP ${resp.status}` }];
    } catch (err) {
        const msg = err instanceof Error ? err.message : "Connection failed";
        return [{ name: "API Connection", ok: false, detail: msg }];
    }
}

/** Detect GPUs via nvidia-smi */
async function checkGpu(): Promise<CheckResult[]> {
    const { execFile } = await import("node:child_process");
    const { promisify } = await import("node:util");
    const exec = promisify(execFile);

    try {
        const { stdout } = await exec("nvidia-smi", [
            "--query-gpu=name,memory.total",
            "--format=csv,noheader",
        ], { timeout: 10_000 });

        const gpus = stdout.trim().split("\n").filter(Boolean);
        if (gpus.length === 0) {
            return [{ name: "GPU Detection", ok: false, detail: "No GPUs found" }];
        }
        return gpus.map((line, i) => ({
            name: `GPU ${i}`,
            ok: true,
            detail: line.trim(),
        }));
    } catch {
        return [{ name: "GPU Detection", ok: false, detail: "nvidia-smi not available" }];
    }
}

/** Run the appropriate check for an auto-check step */
async function runCheck(
    checkId: string,
    answers: Record<string, string | string[]>,
): Promise<CheckResult[]> {
    switch (checkId) {
        case "docker":
            return checkDocker();
        case "api":
            return checkApi((answers["api-url"] as string) || "https://xcelsior.ca");
        case "gpu":
            return checkGpu();
        default:
            return [{ name: checkId, ok: false, detail: "Unknown check" }];
    }
}

export function useWizardFlow(): UseWizardFlowReturn {
    const [stepIndex, setStepIndex] = useState(0);
    const stepIndexRef = useRef(0);
    const [answers, setAnswers] = useState<Record<string, string | string[]>>({});
    const [wizardState, setWizardState] = useState<WizardState>("idle");
    const [wizardMessage, setWizardMessage] = useState(WIZARD_STEPS[0].prompt);
    const [checkResults, setCheckResults] = useState<Record<string, AutoCheckResults>>({});
    const [aiResponse, setAiResponse] = useState<string | null>(null);
    const [aiStreaming, setAiStreaming] = useState(false);
    const [isComplete, setIsComplete] = useState(false);

    const step = WIZARD_STEPS[stepIndex];

    const advanceToNext = useCallback(
        (currentAnswers: Record<string, string | string[]>) => {
            const next = getNextStep(stepIndexRef.current, currentAnswers);
            if (next === -1 || WIZARD_STEPS[next].type === "done") {
                // Find the done step
                const doneIdx = WIZARD_STEPS.findIndex((s) => s.type === "done");
                if (doneIdx >= 0) {
                    setStepIndex(doneIdx);
                    setWizardState("success");
                    setWizardMessage(WIZARD_STEPS[doneIdx].prompt);
                }
                setIsComplete(true);
                return;
            }

            const nextStep = WIZARD_STEPS[next];
            stepIndexRef.current = next;
            setStepIndex(next);
            setWizardState(nextStep.type === "auto-check" ? "thinking" : "idle");
            setWizardMessage(nextStep.prompt);

            // Auto-run checks
            if (nextStep.type === "auto-check" && nextStep.checkId) {
                runCheck(nextStep.checkId, currentAnswers).then((results) => {
                    const allPassed = results.every((r) => r.ok);
                    setCheckResults((prev) => ({
                        ...prev,
                        [nextStep.id]: { items: results, allPassed },
                    }));
                    setWizardState(allPassed ? "success" : "error");
                    setWizardMessage(
                        allPassed
                            ? `All checks passed!`
                            : `${results.filter((r) => !r.ok).length} check(s) failed`,
                    );

                    // Auto-advance after a delay
                    setTimeout(() => {
                        const updatedAnswers = { ...currentAnswers, [nextStep.id]: allPassed ? "passed" : "failed" };
                        setAnswers(updatedAnswers);
                        advanceToNext(updatedAnswers);
                    }, 1500);
                });
            }
        },
        [], // stepIndexRef is stable — no deps needed
    );

    const submitAnswer = useCallback(
        (value: string | string[]) => {
            // Apply defaults
            let resolved = value;
            if (step.id === "api-url" && (value === "" || value === "\r")) {
                resolved = "https://xcelsior.ca";
            }

            const updated = { ...answers, [step.id]: resolved };
            setAnswers(updated);
            advanceToNext(updated);
        },
        [step, answers, advanceToNext],
    );

    const askAi = useCallback(
        async (question: string) => {
            // Build a minimal API config from collected answers
            const baseUrl = (answers["api-url"] as string) || "https://xcelsior.ca";
            const apiKey = (answers["api-key"] as string) || "";

            if (!apiKey) {
                setAiResponse(
                    "I can't connect to the AI assistant yet — complete the API key step first. " +
                    "But here's a tip: check https://xcelsior.ca/dashboard/settings for your API key.",
                );
                return;
            }

            const config: ApiClientConfig = { baseUrl, apiKey, pageContext: `cli-wizard:${step.id}` };

            setAiStreaming(true);
            setAiResponse("");
            setWizardState("thinking");
            setWizardMessage("Xcel is thinking...");

            let content = "";
            try {
                for await (const event of streamChat(config, question)) {
                    if (event.type === "token") {
                        content += event.content ?? "";
                        setAiResponse(content);
                    } else if (event.type === "error") {
                        content = `Error: ${event.message}`;
                        setAiResponse(content);
                    }
                }
                setWizardState("idle");
                setWizardMessage(step.prompt);
            } catch (err) {
                setAiResponse(`Connection error: ${err instanceof Error ? err.message : "unknown"}`);
                setWizardState("error");
                setWizardMessage("AI unavailable — continue with the wizard steps.");
            } finally {
                setAiStreaming(false);
            }
        },
        [answers, step],
    );

    const dismissAi = useCallback(() => {
        setAiResponse(null);
        setWizardState("idle");
        setWizardMessage(step.prompt);
    }, [step]);

    return {
        step,
        stepIndex,
        answers,
        wizardState,
        wizardMessage,
        checkResults,
        aiResponse,
        aiStreaming,
        submitAnswer,
        askAi,
        dismissAi,
        isComplete,
    };
}
