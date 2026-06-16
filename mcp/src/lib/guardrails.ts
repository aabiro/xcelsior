export interface ShouldIRunInput {
  gpu_model: string;
  duration_hours: number;
  spot?: boolean;
  max_hourly_cad?: number;
  require_canada?: boolean;
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

  let jurisdiction_note: string | undefined;
  if (input.require_canada) {
    jurisdiction_note =
      "Canadian data residency is supported — prefer CA regions (e.g. ca-east) when creating instances.";
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