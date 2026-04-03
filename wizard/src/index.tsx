#!/usr/bin/env node
// Xcelsior CLI Wizard — Entry Point
// Structured PostHog-style setup wizard: same steps, same questions, every time.
// The wizard sprite stays on line 1. Steps render below it.
// At any text prompt, type "?" or "help ..." to get an AI answer inline,
// then press Enter to return to the current step.

import React, { useState, useCallback, useMemo } from "react";
import { render, Box, Text, useApp } from "ink";
import { WizardLine, type BranchId } from "./WizardLine.js";
import { setupWizardRegion, resetWizardRegion } from "./useWizardAnimation.js";
import { STATE_COLORS } from "./wizard-sprite.js";
import { WIZARD_STEPS } from "./wizard-flow.js";
import { useWizardFlow } from "./useWizardFlow.js";
import {
  ProgressBar,
  SelectStep,
  TextStep,
  AutoCheckStep,
  ConfirmStep,
  DoneStep,
  AiResponse,
} from "./steps.js";

function App() {
  const { exit } = useApp();
  const [exiting, setExiting] = useState(false);

  const {
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
  } = useWizardFlow();

  const handleExit = useCallback(() => setExiting(true), []);
  const handleExitDone = useCallback(() => {
    resetWizardRegion();
    exit();
  }, [exit]);

  const totalSteps = WIZARD_STEPS.filter(
    (s) => !s.condition || s.condition(answers),
  ).length;
  const currentNum = WIZARD_STEPS.slice(0, stepIndex + 1).filter(
    (s) => !s.condition || s.condition(answers),
  ).length;

  // Map wizard state to branch animation
  const wizardBranch = useMemo((): BranchId | null => {
    if (wizardState === "success" && isComplete) return "celebrate";
    if (wizardState === "success") return "eureka";
    if (wizardState === "error") return "error";
    if (wizardState === "thinking") return "levitate";
    return null;
  }, [wizardState, isComplete]);

  return (
    <Box flexDirection="column">
      {/* Wizard sprite — choreographed animation */}
      <WizardLine
        message={wizardMessage}
        messageColor={STATE_COLORS[wizardState]}
        exiting={exiting}
        onExitDone={handleExitDone}
        branch={wizardBranch}
      />

      {/* Progress bar */}
      {!isComplete && <ProgressBar current={currentNum} total={totalSteps} />}

      {/* AI escape hatch response */}
      {aiResponse !== null && (
        <Box marginTop={1}>
          <AiResponse
            response={aiResponse}
            streaming={aiStreaming}
            onDismiss={dismissAi}
          />
        </Box>
      )}

      {/* Step content (hidden while AI response is showing) */}
      {aiResponse === null && (
        <Box flexDirection="column" marginTop={1}>
          {step.type === "select" && step.options && (
            <SelectStep
              options={step.options}
              onSelect={(v) => submitAnswer(v)}
            />
          )}

          {step.type === "text" && (
            <TextStep
              placeholder={step.placeholder}
              onSubmit={(v) => submitAnswer(v)}
              onAskAi={askAi}
            />
          )}

          {step.type === "auto-check" && (
            <AutoCheckStep results={checkResults[step.id]} />
          )}

          {step.type === "confirm" && (
            <ConfirmStep
              label={step.confirmLabel ?? "Confirm?"}
              onConfirm={(yes) => submitAnswer(yes ? "yes" : "no")}
            />
          )}

          {step.type === "done" && <DoneStep answers={answers} onExit={handleExit} />}
        </Box>
      )}

      {/* Hint: AI escape hatch */}
      {!isComplete && step.type === "text" && aiResponse === null && (
        <Box marginTop={1} marginLeft={4}>
          <Text dimColor>
            Type <Text bold>?</Text> followed by a question to ask Xcel for help
          </Text>
        </Box>
      )}
    </Box>
  );
}

// Set up scroll region: wizard sprite above, Ink content below
setupWizardRegion();

render(<App />);
