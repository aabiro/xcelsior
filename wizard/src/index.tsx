#!/usr/bin/env node
// Xcelsior CLI Wizard — Entry Point
// Structured setup wizard: auth → browse GPUs → pick → configure → pay → launch.
// The wizard sprite (Hexara) stays on line 1. Steps render below it.
// Press "?" at any step to ask Hexara for help inline.

import React, { useState, useCallback, useMemo } from "react";
import { render, Box, Text, useApp, useInput } from "ink";
import { WizardLine, type BranchId } from "./WizardLine.js";
import { setupWizardRegion, resetWizardRegion } from "./useWizardAnimation.js";
import { STATE_COLORS } from "../sprites/wizard/wizard-sprite.js";
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
  ProviderSummaryStep,
  BrandSpinner,
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
    openBrowserNow,
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
    checkAwaitContinue,
    retryCheck,
    skipCheck,
    continueFromCheck,
    continueFromAuth,
    tokenSaveError,
    deviceAuthEnvPath,
    aiAvailable,
    showAiPrompt,
    toggleAiPrompt,
    chatHistory,
    currentAiQuestion,
    providerSummary,
    pendingConfirmation,
    confirmAi,
    aiToolCalls,
    transitioning,
    hasAiDetails,
    revealAi,
  } = useWizardFlow();

  const handleExit = useCallback(() => setExiting(true), []);
  const handleExitDone = useCallback(() => {
    resetWizardRegion();
    exit();
  }, [exit]);

  // Global "?" keybind — opens AI prompt on any step that supports it
  useInput((input, key) => {
    // Quit from anywhere — q exits (except text input & done step which handle q natively)
    if (input === "q" && step.type !== "text" && step.type !== "done" && !showAiPrompt) {
      handleExit();
      return;
    }
    // Ctrl+C also triggers exit animation
    if (key.ctrl && input === "c") {
      handleExit();
      return;
    }
    // AI confirmation handling
    if (pendingConfirmation) {
      if (input === "y" || input === "Y") confirmAi(true);
      if (input === "n" || input === "N") confirmAi(false);
      return;
    }
    if (input === "?" && aiAvailable && aiResponse === null && !showAiPrompt && !aiStreaming) {
      // Only on steps where text input isn't active (text steps handle ? natively)
      if (step.type !== "text" && step.type !== "device-auth" && step.type !== "done") {
        toggleAiPrompt();
      }
    }
    // Reveal buffered AI details
    if ((input === "d" || input === "D") && hasAiDetails && !aiStreaming) {
      revealAi();
      return;
    }
    // Device auth controls — only when still pending/error, not after authorized
    if (step.type === "device-auth" && deviceAuth.status !== "authorized") {
      if ((input === "m" || input === "M") && (deviceAuth.status === "waiting" || deviceAuth.status === "error")) switchToManualAuth();
      if (input === "\r" && deviceAuth.status === "error") retryDeviceAuth();
    }
    // Auto-fetch retry
    if (step.type === "auto-fetch" && browseError && input === "\r") {
      submitAnswer("retry");
    }
  });

  // Compute total steps based only on mode to prevent progress bar jumps
  const totalSteps = useMemo(() => {
    const mode = answers.mode as string | undefined;
    if (!mode) return WIZARD_STEPS.length;
    // Use a stable snapshot with only mode set for condition evaluation
    const stableAnswers: Record<string, string> = { mode };
    return WIZARD_STEPS.filter(
      (s) => !s.condition || s.condition(stableAnswers),
    ).length;
  }, [answers.mode]);
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
    <Box flexDirection="column" alignItems="center">
      {/* Hexara wizard sprite — choreographed animation */}
      <WizardLine
        message={wizardMessage}
        messageColor={STATE_COLORS[wizardState]}
        exiting={exiting}
        onExitDone={handleExitDone}
        branch={wizardBranch}
      />

      {/* Progress bar — hidden until mode is chosen (step count unknown before then) */}
      {!isComplete && answers.mode && <ProgressBar current={currentNum} total={totalSteps} />}

      {/* AI escape hatch response */}
      {aiResponse !== null && (
        <Box marginTop={1} width={64}>
          <AiResponse
            response={aiResponse}
            streaming={aiStreaming}
            onDismiss={dismissAi}
            chatHistory={chatHistory}
            currentQuestion={currentAiQuestion ?? undefined}
            toolCalls={aiToolCalls}
            pendingConfirmation={pendingConfirmation ?? undefined}
          />
        </Box>
      )}

      {/* Inline AI prompt (for non-text steps) */}
      {showAiPrompt && aiResponse === null && (
        <Box marginTop={1} width={64}>
          <AiPrompt
            onSubmit={askAi}
            onCancel={toggleAiPrompt}
          />
        </Box>
      )}

      {/* Spinner during choreography transition */}
      {transitioning && (
        <Box marginTop={1}>
          <BrandSpinner />
        </Box>
      )}

      {/* Step content (hidden while AI response is showing or during transition) */}
      {aiResponse === null && !showAiPrompt && !transitioning && (
        <Box flexDirection="column" marginTop={1} width={64}>
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
              awaitContinue={checkAwaitContinue}
              required={step.checkRequired}
              successTitle={{
                docker: "Docker environment ready!",
                api: "Connection verified!",
                gpu: "GPUs detected!",
                versions: "Version checks passed!",
                benchmark: "Benchmarks complete!",
                network: "Network tests passed!",
                verify: "Hardware verified!",
                "host-register": "Host registered on the marketplace!",
                admission: "Admission checks passed!",
                launch: "Instance launched!",
                wallet: "Wallet check passed!",
              }[step.checkId ?? ""] ?? "All checks passed!"}
              onRetry={retryCheck}
              onSkip={skipCheck}
              onContinue={continueFromCheck}
            />
          )}

          {step.type === "confirm" && step.id === "provider-summary" && providerSummary && (
            <ProviderSummaryStep
              summary={providerSummary}
              onConfirm={(yes) => submitAnswer(yes ? "yes" : "no")}
              error={confirmError}
            />
          )}

          {step.type === "confirm" && (step.id !== "provider-summary" || !providerSummary) && (
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
              onContinue={continueFromAuth}
              onOpenBrowser={openBrowserNow}
              envPath={deviceAuthEnvPath}
              tokenSaveError={tokenSaveError}
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
        <Box marginTop={1} width={64}>
          <Text dimColor>
            {step.type === "text"
              ? <>Type <Text bold>?</Text> followed by a question to ask Hexara for help</>
              : <>Press <Text bold>?</Text> to ask Hexara for help</>
            }
          </Text>
        </Box>
      )}

      {/* Persistent chat history from this step (visible even after dismiss) */}
      {chatHistory.length > 0 && aiResponse === null && !showAiPrompt && (
        <Box marginTop={1} width={64} flexDirection="column" borderStyle="single" borderColor="#374151" paddingX={1}>
          {chatHistory.map((msg, i) => (
            <Box key={i} flexDirection="column">
              <Text color="#00d4ff" dimColor>You: {msg.question}</Text>
              <Text wrap="wrap" dimColor>{msg.answer}</Text>
            </Box>
          ))}
        </Box>
      )}
    </Box>
  );
}

// Set up scroll region: wizard sprite above, Ink content below
setupWizardRegion();

render(<App />);
