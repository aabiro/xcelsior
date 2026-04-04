#!/usr/bin/env node
// Xcelsior CLI Wizard — Entry Point
// Structured setup wizard: auth → browse GPUs → pick → configure → pay → launch.
// The wizard sprite (Hexara) stays on line 1. Steps render below it.
// Press "?" at any step to ask Hexara for help inline.

import React, { useState, useCallback, useMemo } from "react";
import { render, Box, Text, useApp, useInput } from "ink";
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
  AiPrompt,
  DeviceAuthStep,
  ManualTokenStep,
  GpuBrowseStep,
  PaymentGateStep,
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
    deviceAuth,
    switchToManualAuth,
    retryDeviceAuth,
    submitManualToken,
    gpuListings,
    gpuOptions,
    imageOptions,
    instanceInfo,
    validationError,
    confirmError,
    launchSummary,
    paymentGate,
    skipPayment,
    browseError,
    checkCanRetry,
    retryCheck,
    skipCheck,
    aiAvailable,
    showAiPrompt,
    toggleAiPrompt,
  } = useWizardFlow();

  const handleExit = useCallback(() => setExiting(true), []);
  const handleExitDone = useCallback(() => {
    resetWizardRegion();
    exit();
  }, [exit]);

  // Global "?" keybind — opens AI prompt on any step that supports it
  useInput((input, key) => {
    if (input === "?" && aiAvailable && aiResponse === null && !showAiPrompt) {
      // Only on steps where text input isn't active (text steps handle ? natively)
      if (step.type !== "text" && step.type !== "device-auth" && step.type !== "done") {
        toggleAiPrompt();
      }
    }
    // Device auth controls
    if (step.type === "device-auth") {
      if (input === "m" || input === "M") switchToManualAuth();
      if (input === "\r" && deviceAuth.status === "error") retryDeviceAuth();
    }
    // Auto-fetch retry
    if (step.type === "auto-fetch" && browseError && input === "\r") {
      // retrigger browse by re-submitting
      submitAnswer("retry");
    }
  });

  const totalSteps = WIZARD_STEPS.filter(
    (s) => !s.condition || s.condition(answers),
  ).length;
  const currentNum = WIZARD_STEPS.slice(0, stepIndex + 1).filter(
    (s) => !s.condition || s.condition(answers),
  ).length;

  // Map wizard state to branch animation — varied choreography
  const wizardBranch = useMemo((): BranchId | null => {
    if (exiting) return null;
    if (wizardState === "finishing") return "bow";
    if (wizardState === "success" && isComplete) return "celebrate";
    if (wizardState === "excited") return "dance";
    if (wizardState === "success") return "eureka";
    if (wizardState === "error") return "error";
    if (wizardState === "thinking") return "levitate";
    if (wizardState === "waiting") return "sleep";
    return null;
  }, [wizardState, isComplete, exiting]);

  return (
    <Box flexDirection="column">
      {/* Hexara wizard sprite — choreographed animation */}
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

      {/* Inline AI prompt (for non-text steps) */}
      {showAiPrompt && aiResponse === null && (
        <Box marginTop={1}>
          <AiPrompt
            onSubmit={askAi}
            onCancel={toggleAiPrompt}
          />
        </Box>
      )}

      {/* Step content (hidden while AI response is showing) */}
      {aiResponse === null && !showAiPrompt && (
        <Box flexDirection="column" marginTop={1}>
          {step.type === "select" && step.options && (
            <SelectStep
              options={step.id === "gpu-pick" ? gpuOptions : step.id === "image-pick" ? imageOptions : step.options}
              onSelect={(v) => submitAnswer(v)}
            />
          )}

          {step.type === "text" && (
            <TextStep
              placeholder={step.placeholder}
              onSubmit={(v) => submitAnswer(v)}
              onAskAi={aiAvailable ? askAi : undefined}
              validationError={validationError}
            />
          )}

          {step.type === "auto-check" && (
            <AutoCheckStep
              results={checkResults[step.id]}
              canRetry={checkCanRetry}
              required={step.checkRequired}
              onRetry={retryCheck}
              onSkip={skipCheck}
            />
          )}

          {step.type === "confirm" && (
            <ConfirmStep
              label={step.confirmLabel ?? "Confirm?"}
              onConfirm={(yes) => submitAnswer(yes ? "yes" : "no")}
              summary={step.id === "confirm-launch" ? launchSummary : undefined}
              error={confirmError}
            />
          )}

          {step.type === "device-auth" && deviceAuth.status !== "manual" && (
            <DeviceAuthStep
              userCode={deviceAuth.userCode}
              verificationUri={deviceAuth.verificationUri}
              status={deviceAuth.status}
              token={deviceAuth.token ?? undefined}
              email={deviceAuth.email ?? undefined}
              errorMessage={deviceAuth.errorMessage ?? undefined}
            />
          )}

          {step.type === "device-auth" && deviceAuth.status === "manual" && (
            <ManualTokenStep onSubmit={submitManualToken} />
          )}

          {step.type === "auto-fetch" && (
            <GpuBrowseStep
              loading={!browseError && gpuListings.length === 0}
              count={gpuListings.length}
              error={browseError ?? undefined}
            />
          )}

          {step.type === "payment-gate" && (
            <PaymentGateStep
              balance={paymentGate.balance}
              required={paymentGate.required}
              polling={paymentGate.polling}
              billingUrl={paymentGate.billingUrl}
              onSkip={skipPayment}
            />
          )}

          {step.type === "done" && (
            <DoneStep
              answers={answers}
              instanceInfo={instanceInfo ? {
                job_id: instanceInfo.job_id,
                host_ip: instanceInfo.host_ip,
                ssh_port: instanceInfo.ssh_port,
                status: instanceInfo.status,
              } : undefined}
              onExit={handleExit}
            />
          )}
        </Box>
      )}

      {/* Hint: AI escape hatch (visible after auth) */}
      {!isComplete && aiAvailable && step.type !== "device-auth" && aiResponse === null && !showAiPrompt && (
        <Box marginTop={1} marginLeft={4}>
          <Text dimColor>
            {step.type === "text"
              ? <>Type <Text bold>?</Text> followed by a question to ask Hexara for help</>
              : <>Press <Text bold>?</Text> to ask Hexara for help</>
            }
          </Text>
        </Box>
      )}
    </Box>
  );
}

// Set up scroll region: wizard sprite above, Ink content below
setupWizardRegion();

render(<App />);
