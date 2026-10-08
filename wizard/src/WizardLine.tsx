// WizardLine — Choreographed Sixel wizard sprite
// The sprite is drawn directly to stdout above Ink's scroll region.
// This component just manages animation lifecycle and renders the message text.

import React, { useEffect, useRef } from "react";
import { Box, Text } from "ink";
import { useWizardAnimation, type BranchId, type WizardMood } from "./useWizardAnimation.js";

export type { BranchId };

export interface WizardLineProps {
  /** Message text displayed beside the wizard */
  message: string;
  /** Hex color for message text */
  messageColor?: string;
  /** When true, finish current act then play exit sequence */
  exiting?: boolean;
  /** Called when exit animation (outro) completes */
  onExitDone?: () => void;
  /** Trigger a branch animation (eureka, celebrate, error, sleep, levitate, dance, bow) */
  branch?: BranchId | null;
  /** Continuous idle-loop character while no branch is playing */
  mood?: WizardMood;
  /** Re-fire branches when the active step changes (step id). */
  pulseKey?: string;
}

export function WizardLine({ message, messageColor, exiting, onExitDone, branch, mood = "idle", pulseKey }: WizardLineProps) {
  const { done, triggerBranch } = useWizardAnimation(exiting ?? false, mood);

  useEffect(() => {
    if (done && onExitDone) onExitDone();
  }, [done, onExitDone]);

  // Forward branch trigger — branchKey lets the same branch re-fire on step changes.
  const lastBranchKey = useRef<string | null>(null);
  useEffect(() => {
    const key = branch ? `${branch}:${pulseKey ?? ""}:${message.slice(0, 24)}` : null;
    if (branch && key !== lastBranchKey.current) {
      lastBranchKey.current = key;
      triggerBranch(branch);
    } else if (!branch) {
      lastBranchKey.current = null;
    }
  }, [branch, message, pulseKey, triggerBranch]);

  if (done) return null;

  // Message text only — sprite is drawn above Ink's scroll region by useWizardAnimation
  return (
    <Box justifyContent="center">
      <Text color={messageColor}>{message}</Text>
    </Box>
  );
}
