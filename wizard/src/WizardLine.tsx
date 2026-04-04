// WizardLine — Choreographed Sixel wizard sprite
// The sprite is drawn directly to stdout above Ink's scroll region.
// This component just manages animation lifecycle and renders the message text.

import React, { useEffect, useRef } from "react";
import { Box, Text } from "ink";
import { useWizardAnimation, type BranchId } from "./useWizardAnimation.js";

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
}

export function WizardLine({ message, messageColor, exiting, onExitDone, branch }: WizardLineProps) {
  const { done, triggerBranch } = useWizardAnimation(exiting ?? false);

  useEffect(() => {
    if (done && onExitDone) onExitDone();
  }, [done, onExitDone]);

  // Forward branch trigger from props
  const lastBranch = useRef<BranchId | null>(null);
  useEffect(() => {
    if (branch && branch !== lastBranch.current) {
      lastBranch.current = branch;
      triggerBranch(branch);
    } else if (!branch) {
      lastBranch.current = null;
    }
  }, [branch, triggerBranch]);

  if (done) return null;

  // Message text only — sprite is drawn above Ink's scroll region by useWizardAnimation
  return (
    <Box justifyContent="center">
      <Text color={messageColor}>{message}</Text>
    </Box>
  );
}
