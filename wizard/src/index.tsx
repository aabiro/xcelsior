#!/usr/bin/env node
// Xcelsior CLI Wizard — Entry Point
// Structured setup wizard: auth → browse GPUs → pick → configure → pay → launch.
// The wizard sprite (Hexara) stays on line 1. Steps render below it.
// Press "?" at any step to ask Hexara for help inline.

import React, { useState, useCallback, useMemo, useEffect } from "react";
import { pathToFileURL } from "node:url";
import { render, Box, Text, useApp, useInput } from "ink";
import { WizardLine } from "./WizardLine.js";
import { computeWizardBranch, computeWizardMood } from "./hexara-choreography.js";
import { setupWizardRegion, resetWizardRegion, getCustomStdout } from "./useWizardAnimation.js";
import { spriteCapable } from "./capability.js";
import { STATE_COLORS } from "../sprites/wizard/wizard-sprite.js";
import { WIZARD_STEPS, STATIC_STEP_HELP } from "./wizard-flow.js";
import { useWizardFlow } from "./useWizardFlow.js";
import { StatusGate } from "./StatusGate.js";
import { WorkScreen } from "./WorkScreen.js";
import { buildTaskList, computeTaskStates } from "./task-model.js";
import { timeHintForStep, buildMarketplaceSlides, LEARN_SLIDES, sdkLearnSlides } from "./learn-content.js";
import { TitleBar, KeybindFooter } from "./chrome.js";
import { StartupCard } from "./StartupCard.js";
import { detectEnvironment } from "./environment.js";
import { initSpriteCapability } from "./capability.js";
import {
  ProgressBar,
  SelectStep,
  TextStep,
  AutoCheckStep,
  ConfirmStep,
  SdkSnippetStep,
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

// Long-running step types where the two-pane WorkScreen (Learn + Tasks) shows.
const WORK_STEP_TYPES = ["auto-check", "auto-fetch", "payment-gate"];

export function App() {
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
    resumeInfo,
    flushCheckpoint,
    gatePhase,
    serviceStatus,
    proceedFromGate,
    recheckGate,
    continueAnywayFromGate,
    resumedStepLabel,
    checkProgress,
    marketplaceStats,
  } = useWizardFlow();

  // Prefer live marketplace charts in the Learn pane; fall back to concept cards.
  const learnSlides = useMemo(() => {
    if (answers.mode === "sdk") return sdkLearnSlides();
    if (!marketplaceStats) return undefined;
    const live = buildMarketplaceSlides(marketplaceStats);
    if (live.length === 0) return undefined;
    return [...live, ...LEARN_SLIDES.filter((s) => s.mode === "learn")];
  }, [answers.mode, marketplaceStats]);

  const gateOpen = gatePhase !== "passed" && !isComplete;
  const statusUrl = `${(answers["_api_base_url"] as string) || "https://xcelsior.ca"}/dashboard`;
  const env = useMemo(() => detectEnvironment(), []);
  // When Hexara is unavailable, surface static per-step help (Part A). Keyed by
  // step id, falling back to checkId (STATIC_STEP_HELP uses checkIds like "versions").
  const staticHelp = !aiAvailable
    ? (STATIC_STEP_HELP[step.id] ?? STATIC_STEP_HELP[step.checkId ?? ""])
    : undefined;

  // Live task checklist (Part D) — derived as a view over the real flow state.
  const tasks = useMemo(() => {
    const mode = answers.mode as string | undefined;
    const list = buildTaskList(mode);
    if (list.length === 0) return list;
    const completedStepIds = WIZARD_STEPS.slice(0, stepIndex).map((s) => s.id);
    return computeTaskStates(list, {
      currentStepId: step.id,
      completedStepIds,
      checkResults,
      currentFailed: checkCanRetry,
    });
  }, [answers.mode, stepIndex, step.id, checkResults, checkCanRetry]);

  // Rolling status log of granular sub-actions (Part E) — sourced from the
  // wizard's own message stream, deduped so repeats don't spam the strip.
  const [statusLog, setStatusLog] = useState<string[]>([]);
  useEffect(() => {
    if (!wizardMessage) return;
    setStatusLog((prev) => (prev[prev.length - 1] === wizardMessage ? prev : [...prev, wizardMessage].slice(-12)));
  }, [wizardMessage]);

  // Two-pane WorkScreen is shown only on long-running steps with latency to fill.
  const isWorkStep =
    !gateOpen &&
    !isComplete &&
    !!answers.mode &&
    WORK_STEP_TYPES.includes(step.type) &&
    aiResponse === null &&
    !showAiPrompt &&
    !transitioning;

  const handleExit = useCallback(() => {
    flushCheckpoint();
    setExiting(true);
  }, [flushCheckpoint]);
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
    // Interactive nudge: "Press A to ask Hexara" during long-running work steps
    if ((input === "a" || input === "A") && aiAvailable && aiResponse === null && !showAiPrompt && !aiStreaming
      && WORK_STEP_TYPES.includes(step.type)) {
      toggleAiPrompt();
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

  const hexaraCtx = useMemo(() => ({
    exiting,
    isComplete,
    wizardState,
    step,
    mode: answers.mode as string | undefined,
    transitioning,
    aiStreaming,
    showAiPrompt,
    aiResponseOpen: aiResponse !== null,
    checkResults,
    browseError,
    deviceAuthStatus: deviceAuth.status,
    gateOpen,
    stepPulseKey: step.id,
  }), [
    exiting, isComplete, wizardState, step, answers.mode, transitioning,
    aiStreaming, showAiPrompt, aiResponse, checkResults, browseError,
    deviceAuth.status, gateOpen,
  ]);

  const wizardMood = useMemo(() => computeWizardMood(hexaraCtx), [hexaraCtx]);
  const wizardBranch = useMemo(() => computeWizardBranch(hexaraCtx), [hexaraCtx]);

  return (
    <Box flexDirection="column" alignItems="center" paddingRight={spriteCapable() ? 42 : 0} width="100%">
      {/* Persistent branded title bar (Part I) */}
      <TitleBar />

      {/* Hexara wizard sprite — choreographed animation */}
      <WizardLine
        message={wizardMessage}
        messageColor={STATE_COLORS[wizardState]}
        exiting={exiting}
        onExitDone={handleExitDone}
        branch={wizardBranch}
        mood={wizardMood}
        pulseKey={step.id}
      />

      {/* Resume / expiry notices */}
      {resumeInfo.resumed && !isComplete && (
        <Box marginTop={1} width={64}>
          <Text color="#fbbf24">
            ↻ Resumed previous session{resumedStepLabel ? <Text> at: <Text bold>{resumedStepLabel}</Text></Text> : null}
          </Text>
        </Box>
      )}
      {resumeInfo.needsReauth && !isComplete && (
        <Box marginTop={1} width={64}>
          <Text color="#fbbf24">Sign in again to continue where you left off</Text>
        </Box>
      )}

      {/* Opening: startup card while checking; gate panel when degraded/blocked */}
      {gateOpen && (
        <Box marginTop={1}>
          {gatePhase === "checking" ? (
            <StartupCard env={env} />
          ) : (
            <StatusGate
              phase={gatePhase}
              report={serviceStatus}
              statusUrl={statusUrl}
              onProceed={proceedFromGate}
              onRecheck={recheckGate}
              onContinueAnyway={continueAnywayFromGate}
            />
          )}
        </Box>
      )}

      {!gateOpen && (<>
      {/* Progress bar — hidden until mode is chosen, and replaced by the task pane on work steps */}
      {!isComplete && answers.mode && !isWorkStep && <ProgressBar current={currentNum} total={totalSteps} />}

      {/* Two-pane Learn + Tasks layout on long-running steps (Part B) */}
      {isWorkStep && (
        <Box marginTop={1}>
          <WorkScreen
            tasks={tasks}
            slides={learnSlides}
            log={statusLog}
            tail={checkProgress}
            intro={timeHintForStep(step.id)}
            nudge={aiAvailable ? "Press A to ask Hexara anything while you wait" : null}
            aiAvailable={aiAvailable}
          />
        </Box>
      )}

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
              staticHelp={staticHelp}
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
                "sdk-detect": "Project detected!",
                "sdk-install": "SDK package ready!",
                "sdk-credentials": "Credentials configured!",
                "sdk-verify": "API connection verified!",
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

          {step.type === "confirm" && step.id === "sdk-snippet" && (
            <SdkSnippetStep
              snippet={(answers["_sdk_snippet"] as string) || ""}
              envPath={(answers["_sdk_env_path"] as string) || ".env.local"}
              onConfirm={() => submitAnswer("yes")}
            />
          )}

          {step.type === "confirm" && step.id !== "sdk-snippet" && (step.id !== "provider-summary" || !providerSummary) && (
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
              oauthClientId={answers["oauth-client-id"] as string | undefined}
              oauthClientSecret={answers["oauth-client-secret"] as string | undefined}
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
      </>)}

      {/* Context-sensitive keybind footer (Part I) */}
      <Box marginTop={1}>
        <KeybindFooter
          ctx={{
            gatePhase,
            stepType: step.type,
            isComplete,
            aiAvailable,
            isWorkStep,
            canRetry: checkCanRetry,
            awaitContinue: checkAwaitContinue,
            showAiPrompt,
            aiResponseOpen: aiResponse !== null,
          }}
        />
      </Box>
    </Box>
  );
}

// Bootstrap only when run as the CLI entry — not when imported by tests.
function isEntryModule(): boolean {
  try {
    const entry = process.argv[1];
    if (!entry) return false;
    return import.meta.url === pathToFileURL(entry).href;
  } catch {
    return false;
  }
}

if (isEntryModule()) {
  void (async () => {
    // Probe the terminal for Sixel support (DA1) before Ink takes over stdin, so
    // the sprite only paints where it'll render cleanly. Falls back fast on
    // non-TTY / opt-out, so this never hangs headless.
    await initSpriteCapability();

    // Set up stream interceptor for flawless Hexara Sixel rendering
    setupWizardRegion();

    render(<App />, { stdout: getCustomStdout() });
  })();
}
