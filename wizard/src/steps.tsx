// Xcelsior Wizard — Step Renderers
// Pure display components for each step type in the structured flow.
// No state management — all state lives in useWizardFlow.

import React, { useState, useCallback } from "react";
import { Text, Box, useInput } from "ink";
import SelectInput from "ink-select-input";
import TextInput from "ink-text-input";
import type { WizardStep, SelectOption } from "./wizard-flow.js";
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
    <Text dimColor>
      {"  "}[{"█".repeat(filled)}{"░".repeat(empty)}] {current}/{total}
    </Text>
  );
}

// ── Select step ──────────────────────────────────────────────────────

interface SelectStepProps {
  options: SelectOption[];
  onSelect: (value: string) => void;
}

export function SelectStep({ options, onSelect }: SelectStepProps) {
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
  onAskAi: (question: string) => void;
}

export function TextStep({ placeholder, onSubmit, onAskAi }: TextStepProps) {
  const [value, setValue] = useState("");

  const handleSubmit = useCallback(
    (text: string) => {
      const trimmed = text.trim();
      // Detect questions — route to AI escape hatch
      if (trimmed.startsWith("?") || trimmed.toLowerCase().startsWith("help")) {
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
    <Box marginLeft={4}>
      <Text color="#00d4ff">{"› "}</Text>
      <TextInput
        value={value}
        onChange={setValue}
        onSubmit={handleSubmit}
        placeholder={placeholder}
      />
    </Box>
  );
}

// ── Auto-check step (results display) ────────────────────────────────

interface AutoCheckStepProps {
  results?: AutoCheckResults;
}

export function AutoCheckStep({ results }: AutoCheckStepProps) {
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
    </Box>
  );
}

// ── Confirm step ─────────────────────────────────────────────────────

interface ConfirmStepProps {
  label: string;
  onConfirm: (yes: boolean) => void;
}

export function ConfirmStep({ label, onConfirm }: ConfirmStepProps) {
  useInput((input) => {
    if (input === "y" || input === "Y" || input === "\r") onConfirm(true);
    if (input === "n" || input === "N") onConfirm(false);
  });

  return (
    <Box marginLeft={4} flexDirection="column">
      <Text>{label}</Text>
      <Text dimColor>Press <Text bold>y</Text> to confirm, <Text bold>n</Text> to skip</Text>
    </Box>
  );
}

// ── Done step ────────────────────────────────────────────────────────

interface DoneStepProps {
  answers: Record<string, string | string[]>;
  /** Called when user presses Enter/q — triggers wizard exit animation */
  onExit: () => void;
}

export function DoneStep({ answers, onExit }: DoneStepProps) {
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
      {answers["api-url"] && (
        <Text>  API: <Text bold>{answers["api-url"] as string}</Text></Text>
      )}
      <Text />
      <Text color="#00d4ff">Next steps:</Text>
      {(mode === "rent" || mode === "both") && (
        <Text>  • Browse GPUs: <Text bold>xcelsior marketplace</Text></Text>
      )}
      {(mode === "provide" || mode === "both") && (
        <Text>  • Register host: <Text bold>xcelsior host-add</Text></Text>
      )}
      <Text>  • AI assistant: <Text bold>xcelsior ai</Text></Text>
      <Text />
      <Text dimColor>Press Enter or q to exit</Text>
    </Box>
  );
}

// ── AI Response Inline ───────────────────────────────────────────────

interface AiResponseProps {
  response: string;
  streaming: boolean;
  onDismiss: () => void;
}

export function AiResponse({ response, streaming, onDismiss }: AiResponseProps) {
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
      borderColor="#00d4ff"
      paddingX={1}
      paddingY={0}
    >
      <Text color="#00d4ff" bold>Xcel says:</Text>
      <Text wrap="wrap">{response}</Text>
      {!streaming && (
        <Text dimColor>Press Enter to continue with setup</Text>
      )}
    </Box>
  );
}
