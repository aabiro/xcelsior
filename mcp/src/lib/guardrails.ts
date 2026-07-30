export interface ShouldIRunInput {
  gpu_model: string;
  duration_hours: number;
  spot?: boolean;
  max_hourly_cad?: number;
  /** Region or jurisdiction code the workload must stay within. Omitted when unconstrained. */
  require_residency?: string;
}

export interface ShouldIRunResult {
  approved: boolean;
  reasons: string[];
  estimated_cost: Record<string, unknown>;
  wallet_balance_cad: number;
  hourly_rate_cad: number;
  jurisdiction_note?: string;
}

export function evaluateShouldIRunThis(
  input: ShouldIRunInput,
  estimate: Record<string, unknown>,
  walletBalanceCad: number,
): ShouldIRunResult {
  const reasons: string[] = [];
  const hourly =
    Number(estimate.rate_cad_per_hour) ||
    (Number(estimate.gross_cost_cad) > 0 && input.duration_hours > 0
      ? Number(estimate.gross_cost_cad) / input.duration_hours
      : 0);
  const gross = Number(estimate.gross_cost_cad) || 0;

  if (input.max_hourly_cad !== undefined && hourly > input.max_hourly_cad) {
    reasons.push(
      `Hourly rate $${hourly.toFixed(2)} CAD exceeds max_hourly_cad $${input.max_hourly_cad.toFixed(2)}`,
    );
  }

  if (walletBalanceCad <= 0) {
    reasons.push("Wallet balance is zero or negative — add funds before launching");
  } else if (gross > 0 && walletBalanceCad < gross) {
    reasons.push(
      `Estimated ${input.duration_hours}h cost $${gross.toFixed(2)} CAD exceeds wallet balance $${walletBalanceCad.toFixed(2)}`,
    );
  }

  // Residency is a per-workload constraint, not a platform default. Report it back as an explicit
  // instruction to verify, rather than asserting that any particular region satisfies it.
  let jurisdiction_note: string | undefined;
  const residency = input.require_residency?.trim();
  if (residency) {
    jurisdiction_note =
      `This workload requires ${residency} residency. Pass that region constraint explicitly to ` +
      `create_instance and confirm the selected host reports a matching jurisdiction before launching. ` +
      `Do not assume the default region satisfies it.`;
  }

  return {
    approved: reasons.length === 0,
    reasons,
    estimated_cost: estimate,
    wallet_balance_cad: walletBalanceCad,
    hourly_rate_cad: hourly,
    jurisdiction_note,
  };
}