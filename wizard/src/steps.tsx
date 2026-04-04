// Xcelsior Wizard — Step Renderers
// Pure display components for each step type in the structured flow.
// No state management — all state lives in useWizardFlow.

import React, { useState, useCallback, useEffect, useRef } from "react";
import { Text, Box, useInput } from "ink";
import SelectInput from "ink-select-input";
import TextInput from "ink-text-input";
import type { SelectOption } from "./wizard-flow.js";
import type { AutoCheckResults } from "./useWizardFlow.js";

// ── Progress bar ─────────────────────────────────────────────────────

interface ProgressProps {
  current: number;
  total: number;
}

export function ProgressBar({ current, total }: ProgressProps) {
  const filled = Math.round((current / total) * 20);
  const empty = 20 - filled;
  return (
    <Box justifyContent="center" marginTop={1}>
      <Text dimColor>
        [{"█".repeat(filled)}{"░".repeat(empty)}] {current}/{total}
      </Text>
    </Box>
  );
}

// ── Select step ──────────────────────────────────────────────────────

interface SelectStepProps {
  options: SelectOption[];
  onSelect: (value: string) => void;
}

export function SelectStep({ options, onSelect }: SelectStepProps) {
  if (!options || options.length === 0) {
    return (
      <Box marginLeft={4}>
        <Text dimColor>No options available. Press <Text bold>Enter</Text> to go back.</Text>
      </Box>
    );
  }
  return (
    <Box marginLeft={4}>
      <SelectInput
        items={options}
        onSelect={(item) => onSelect(item.value)}
      />
    </Box>
  );
}

// ── Text input step ──────────────────────────────────────────────────

interface TextStepProps {
  placeholder?: string;
  onSubmit: (value: string) => void;
  /** If user types a question (starts with ? or "help"), route to AI */
  onAskAi?: (question: string) => void;
  /** Validation error from the last submission */
  validationError?: string | null;
}

export function TextStep({ placeholder, onSubmit, onAskAi, validationError }: TextStepProps) {
  const [value, setValue] = useState("");

  const handleSubmit = useCallback(
    (text: string) => {
      const trimmed = text.trim();
      // Detect questions — route to AI escape hatch
      if (onAskAi && (trimmed.startsWith("?") || trimmed.toLowerCase().startsWith("help"))) {
        onAskAi(trimmed.replace(/^\?/, "").trim());
        setValue("");
        return;
      }
      onSubmit(trimmed);
      setValue("");
    },
    [onSubmit, onAskAi],
  );

  return (
    <Box marginLeft={4} flexDirection="column">
      <Box>
        <Text color="#00d4ff">{"› "}</Text>
        <TextInput
          value={value}
          onChange={setValue}
          onSubmit={handleSubmit}
          placeholder={placeholder}
        />
      </Box>
      {validationError && (
        <Text color="#ef4444">  ⚠ {validationError}</Text>
      )}
    </Box>
  );
}

// ── Auto-check step (results display) ────────────────────────────────

interface AutoCheckStepProps {
  results?: AutoCheckResults;
  /** When true, checks failed and user can retry or skip */
  canRetry?: boolean;
  /** When true, user cannot skip (must retry or fix) */
  required?: boolean;
  onRetry?: () => void;
  onSkip?: () => void;
}

export function AutoCheckStep({ results, canRetry, required, onRetry, onSkip }: AutoCheckStepProps) {
  useInput((input) => {
    if (!canRetry) return;
    if (input === "\r" || input === "r" || input === "R") onRetry?.();
    if (!required && (input === "s" || input === "S")) onSkip?.();
  });

  if (!results) {
    return (
      <Box marginLeft={4}>
        <Text dimColor>Running checks...</Text>
      </Box>
    );
  }

  return (
    <Box flexDirection="column" marginLeft={4}>
      {results.items.map((r) => (
        <Text key={r.name}>
          <Text color={r.ok ? "#22c55e" : "#ef4444"}>
            {r.ok ? "✓" : "✗"}
          </Text>
          <Text> {r.name}: {r.detail}</Text>
        </Text>
      ))}
      {canRetry && (
        <Box marginTop={1}>
          <Text dimColor>
            Press <Text bold>Enter</Text> to retry
            {!required && <Text>, <Text bold>s</Text> to skip</Text>}
          </Text>
        </Box>
      )}
    </Box>
  );
}

// ── Confirm step ─────────────────────────────────────────────────────

interface ConfirmStepProps {
  label: string;
  onConfirm: (yes: boolean) => void;
  /** Show a summary of what's being confirmed */
  summary?: string[];
  /** Validation error on bad key press */
  error?: string | null;
}

export function ConfirmStep({ label, onConfirm, summary, error }: ConfirmStepProps) {
  useInput((input) => {
    if (input === "y" || input === "Y") onConfirm(true);
    else if (input === "n" || input === "N") onConfirm(false);
    // All other keys including Enter are ignored — validation in useWizardFlow
  });

  return (
    <Box marginLeft={4} flexDirection="column">
      {summary && summary.length > 0 && (
        <Box flexDirection="column" marginBottom={1}>
          {summary.map((line, i) => (
            <Text key={i} dimColor>  {line}</Text>
          ))}
        </Box>
      )}
      <Text>{label}</Text>
      <Text dimColor>Press <Text bold>y</Text> to confirm, <Text bold>n</Text> to cancel</Text>
      {error && <Text color="#ef4444">  ⚠ {error}</Text>}
    </Box>
  );
}

// ── Device Auth step ─────────────────────────────────────────────────

interface DeviceAuthStepProps {
  userCode: string | null;
  verificationUri: string | null;
  status: "loading" | "waiting" | "authorized" | "error" | "manual";
  token?: string;
  email?: string;
  errorMessage?: string;
  onManualFallback?: () => void;
}

export function DeviceAuthStep({
  userCode, verificationUri, status, token, email, errorMessage,
}: DeviceAuthStepProps) {
  const [countdown, setCountdown] = useState<number | null>(null);

  // Start 10s countdown when we get the verification URI and are waiting
  useEffect(() => {
    if (status === "waiting" && verificationUri && countdown === null) {
      setCountdown(10);
    }
  }, [status, verificationUri]);

  // Tick the countdown
  useEffect(() => {
    if (countdown === null || countdown <= 0) return;
    const timer = setTimeout(() => setCountdown(countdown - 1), 1000);
    return () => clearTimeout(timer);
  }, [countdown]);

  if (status === "loading") {
    return (
      <Box marginLeft={4}>
        <Text dimColor>Initiating authentication...</Text>
      </Box>
    );
  }

  if (status === "error") {
    return (
      <Box marginLeft={4} flexDirection="column">
        <Text color="#ef4444">✗ {errorMessage ?? "Authentication failed"}</Text>
        <Text dimColor>Press <Text bold>Enter</Text> to retry, <Text bold>m</Text> for manual token paste</Text>
      </Box>
    );
  }

  if (status === "authorized") {
    return (
      <Box marginLeft={4} flexDirection="column">
        <Text color="#22c55e">✓ Authenticated{email ? ` as ${email}` : ""}</Text>
        {token && (
          <Box marginTop={1} flexDirection="column">
            <Text>Your API key:</Text>
            <Box marginTop={0}>
              <Text bold color="#ffcc00">{token}</Text>
            </Box>
            <Text dimColor>Saved to <Text bold>~/.xcelsior/token.json</Text> and <Text bold>~/.xcelsior/config.toml</Text></Text>
          </Box>
        )}
        <Box marginTop={1}>
          <Text dimColor italic>Continuing automatically...</Text>
        </Box>
      </Box>
    );
  }

  if (status === "manual") {
    return (
      <Box marginLeft={4} flexDirection="column">
        <Text dimColor>Manual authentication — paste your API token below</Text>
      </Box>
    );
  }

  // status === "waiting"
  return (
    <Box marginLeft={4} flexDirection="column">
      {verificationUri && countdown !== null && countdown > 0 && (
        <Text>Opening browser in <Text bold color="#ffcc00">{countdown}s</Text> → <Text bold color="#00d4ff">{verificationUri}</Text></Text>
      )}
      {verificationUri && (countdown === null || countdown <= 0) && (
        <Text>Browser opened → <Text bold color="#00d4ff">{verificationUri}</Text></Text>
      )}
      {userCode && (
        <Box marginTop={1}>
          <Text>Enter this code: </Text>
          <Text bold color="#ffcc00" inverse>{` ${userCode} `}</Text>
        </Box>
      )}
      <Box marginTop={1}>
        <Text dimColor>Waiting for authorization... (press <Text bold>m</Text> for manual paste)</Text>
      </Box>
    </Box>
  );
}

// ── Manual token input (fallback for device-auth) ────────────────────

interface ManualTokenStepProps {
  onSubmit: (token: string) => void;
}

export function ManualTokenStep({ onSubmit }: ManualTokenStepProps) {
  const [value, setValue] = useState("");

  const handleSubmit = useCallback(
    (text: string) => {
      const trimmed = text.trim();
      if (!trimmed) return; // don't accept empty
      onSubmit(trimmed);
      setValue("");
    },
    [onSubmit],
  );

  return (
    <Box marginLeft={4} flexDirection="column">
      <Text dimColor>Paste your API token (from Dashboard → Settings → API Keys):</Text>
      <Box>
        <Text color="#00d4ff">{"› "}</Text>
        <TextInput
          value={value}
          onChange={setValue}
          onSubmit={handleSubmit}
          placeholder="xc-..."
        />
      </Box>
    </Box>
  );
}

// ── GPU browser / marketplace results ────────────────────────────────

interface GpuBrowseStepProps {
  loading: boolean;
  count?: number;
  error?: string;
}

export function GpuBrowseStep({ loading, count, error }: GpuBrowseStepProps) {
  if (loading) {
    return (
      <Box marginLeft={4}>
        <Text dimColor>Searching the marketplace...</Text>
      </Box>
    );
  }
  if (error) {
    return (
      <Box marginLeft={4} flexDirection="column">
        <Text color="#ef4444">✗ {error}</Text>
        <Text dimColor>Press <Text bold>Enter</Text> to retry</Text>
      </Box>
    );
  }
  return (
    <Box marginLeft={4}>
      <Text color="#22c55e">✓ Found {count ?? 0} available GPU(s)</Text>
    </Box>
  );
}

// ── Payment gate step ────────────────────────────────────────────────

interface PaymentGateStepProps {
  balance: number;
  required: number;
  polling: boolean;
  billingUrl: string;
  onSkip: () => void;
}

export function PaymentGateStep({ balance, required, polling, billingUrl, onSkip }: PaymentGateStepProps) {
  useInput((input) => {
    if (input === "s" || input === "S") onSkip();
  });

  return (
    <Box marginLeft={4} flexDirection="column">
      <Text>Balance: <Text bold color="#ffcc00">${balance.toFixed(2)} CAD</Text> — need at least <Text bold>${required.toFixed(2)}/hr</Text></Text>
      <Box marginTop={1}>
        <Text>Add funds → <Text bold color="#00d4ff">{billingUrl}</Text></Text>
      </Box>
      {polling && (
        <Text dimColor>Checking for deposit...</Text>
      )}
      <Box marginTop={1}>
        <Text dimColor>Press <Text bold>s</Text> to skip (instance may stop if balance runs out)</Text>
      </Box>
    </Box>
  );
}

// ── Done step ────────────────────────────────────────────────────────

interface DoneStepProps {
  answers: Record<string, string | string[]>;
  instanceInfo?: {
    job_id: string;
    host_ip?: string;
    ssh_port?: number;
    status: string;
  };
  /** Called when user presses Enter/q — triggers wizard exit animation */
  onExit: () => void;
}

export function DoneStep({ answers, instanceInfo, onExit }: DoneStepProps) {
  const mode = answers.mode as string;

  useInput((input) => {
    if (input === "q" || input === "\r") {
      onExit();
    }
  });

  const modeLabel =
    mode === "rent" ? "GPU Renter" : mode === "provide" ? "GPU Provider" : "Renter + Provider";

  return (
    <Box marginLeft={4} flexDirection="column">
      <Text color="#22c55e" bold>Configuration saved!</Text>
      <Text>  Mode: <Text bold>{modeLabel}</Text></Text>

      {instanceInfo && (
        <Box flexDirection="column" marginTop={1}>
          <Text color="#22c55e" bold>Instance launched!</Text>
          <Text>  Status: <Text bold>{instanceInfo.status}</Text> (may take a few minutes to spin up)</Text>
          {instanceInfo.host_ip && instanceInfo.ssh_port && (
            <Text>  SSH: <Text bold>ssh -p {instanceInfo.ssh_port} user@{instanceInfo.host_ip}</Text></Text>
          )}
          <Text>  Dashboard: <Text bold color="#00d4ff">https://xcelsior.ca/dashboard/instances/{instanceInfo.job_id}</Text></Text>
        </Box>
      )}

      {!instanceInfo && (
        <Box flexDirection="column" marginTop={1}>
          <Text color="#00d4ff">Next steps:</Text>
          {(mode === "rent" || mode === "both") && (
            <Text>  • Browse GPUs: <Text bold>xcelsior marketplace</Text></Text>
          )}
          {(mode === "provide" || mode === "both") && (
            <Text>  • Register host: <Text bold>xcelsior host-add</Text></Text>
          )}
          <Text>  • AI assistant: <Text bold>xcelsior ai</Text></Text>
        </Box>
      )}

      <Box marginTop={1}>
        <Text color="#a78bfa" italic>Hexara will be here whenever you need — just run `xcelsior setup` to summon the wizard again.</Text>
      </Box>
      <Text />
      <Text dimColor>Press Enter or q to exit</Text>
    </Box>
  );
}

// ── AI Response Inline (with chat history) ──────────────────────────

interface ChatMessage {
  question: string;
  answer: string;
}

interface AiResponseProps {
  response: string;
  streaming: boolean;
  onDismiss: () => void;
  /** Previous Q&A pairs from this step's session */
  chatHistory?: ChatMessage[];
  /** The current question being answered */
  currentQuestion?: string;
}

export function AiResponse({ response, streaming, onDismiss, chatHistory, currentQuestion }: AiResponseProps) {
  useInput((input) => {
    if (!streaming && (input === "\r" || input === " ")) {
      onDismiss();
    }
  });

  return (
    <Box
      flexDirection="column"
      marginLeft={4}
      borderStyle="round"
      borderColor="#a78bfa"
      paddingX={1}
      paddingY={0}
    >
      <Text color="#a78bfa" bold>Hexara</Text>
      {/* Previous chat history */}
      {chatHistory && chatHistory.map((msg, i) => (
        <Box key={i} flexDirection="column" marginBottom={1}>
          <Text color="#00d4ff" dimColor>  You: {msg.question}</Text>
          <Text wrap="wrap">  {msg.answer}</Text>
        </Box>
      ))}
      {/* Current question */}
      {currentQuestion && (
        <Text color="#00d4ff" dimColor>  You: {currentQuestion}</Text>
      )}
      {/* Current response */}
      <Text wrap="wrap">  {response}</Text>
      {!streaming && (
        <Box marginTop={1}>
          <Text dimColor>Press <Text bold>Enter</Text> to continue · <Text bold>?</Text> to ask another question</Text>
        </Box>
      )}
    </Box>
  );
}

// ── Inline AI prompt (for non-text steps) ───────────────────────────

interface AiPromptProps {
  onSubmit: (question: string) => void;
  onCancel: () => void;
}

export function AiPrompt({ onSubmit, onCancel }: AiPromptProps) {
  const [value, setValue] = useState("");

  useInput((_input, key) => {
    if (key.escape) onCancel();
  });

  const handleSubmit = useCallback(
    (text: string) => {
      const trimmed = text.trim();
      if (!trimmed) {
        onCancel();
        return;
      }
      onSubmit(trimmed);
      setValue("");
    },
    [onSubmit, onCancel],
  );

  return (
    <Box marginLeft={4}>
      <Text color="#a78bfa">Ask Hexara: </Text>
      <TextInput
        value={value}
        onChange={setValue}
        onSubmit={handleSubmit}
        placeholder="Type your question..."
      />
    </Box>
  );
}

// ── Project detection step ───────────────────────────────────────────

interface ProjectDetectStepProps {
  framework: string | null;
  envPath: string | null;
  onConfirm: (yes: boolean) => void;
}

export function ProjectDetectStep({ framework, envPath, onConfirm }: ProjectDetectStepProps) {
  useInput((input) => {
    if (input === "y" || input === "Y") onConfirm(true);
    else if (input === "n" || input === "N") onConfirm(false);
  });

  if (!framework) return null;

  return (
    <Box marginLeft={4} flexDirection="column">
      <Text color="#22c55e">Detected <Text bold>{framework}</Text> project</Text>
      <Text>Save API token to <Text bold>{envPath ?? ".env"}</Text>?</Text>
      <Text dimColor>Press <Text bold>y</Text> to save, <Text bold>n</Text> to skip</Text>
    </Box>
  );
}
