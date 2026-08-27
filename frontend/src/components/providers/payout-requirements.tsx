"use client";

/**
 * Why this provider is not being paid — the same answer the agent gets.
 *
 * Gate P6's frontend clause: *"Provider dashboard parity: the same earnings and
 * payout state the tools return."* Earnings already matched, field for field.
 * Payout state did not: `GET /api/providers/{id}` returns a `payouts` object —
 * `charges_enabled`, `payouts_enabled`, `currently_due`, `past_due`,
 * `disabled_reason`, `checked_live` — `get_provider_account` hands the whole
 * thing to the model, and the dashboard's type dropped it. The browser rendered
 * `status` alone.
 *
 * That is the exact insufficiency `stripe_connect.get_provider` was changed to
 * fix, in its own words:
 *
 * > *`status` alone collapses "we need your bank account" and "we need a photo
 * > of your ID" into `restricted`, which tells a provider nothing they can act
 * > on — and the data to say more is already in the response, it was just being
 * > dropped after deriving the status from it.*
 *
 * It was fixed for the agent and left broken for the person. A provider asking
 * their agent got a field list; the same provider on the dashboard got the word
 * "Restricted".
 *
 * ## `checked_live` decides everything
 *
 * When Stripe is unconfigured, the account has no id, or the retrieve threw,
 * every other field is its zero value — and an empty `currently_due` then reads
 * as "nothing outstanding, you are done" on the one surface whose job is to
 * explain why someone is unpaid. So `checked_live: false` renders as *unknown*,
 * never as *ready* and never as *nothing to do*.
 *
 * ## The glossary annotates; it does not gate
 *
 * The first version derived a label mechanically — `external_account` →
 * "External account" — which is not an explanation, it is the same token with
 * different spacing, dressed up to look translated. Worse, it would have made
 * the requirement *look* handled and discouraged anyone from writing the real
 * thing. A provider reading "External account" learns nothing they did not
 * already know from `external_account`.
 *
 * So: a real sentence for the handful of requirements that actually occur, and
 * for anything else the **raw Stripe name, unchanged** — not a re-spaced
 * imitation of a sentence. A reader can tell the two apart, which is the point.
 *
 * This is a hand-maintained map of a vocabulary Stripe owns, which is the shape
 * that went wrong with `min_tier`. The difference is that the `min_tier` map
 * **gated** — an unlisted tier was refused, so the map's incompleteness became
 * a rejection. This one only annotates: an unlisted requirement renders in
 * full, is never skipped, and still blocks payouts exactly as it should. An
 * incomplete glossary costs a sentence, not a behaviour.
 *
 * The raw name is always shown alongside, because it is what the agent returns
 * — which is what makes the parity literal rather than approximate.
 */

import { AlertTriangle, CheckCircle2, HelpCircle } from "lucide-react";
import type { ProviderPayoutState } from "@/lib/api";

/**
 * What Stripe is actually asking for, for the requirements that actually occur.
 *
 * Deliberately short. Every entry here is a sentence someone can act on; the
 * moment one becomes a restatement of its own key it should be deleted rather
 * than kept for coverage.
 */
const REQUIREMENT_GLOSSARY: Record<string, string> = {
  external_account: "A bank account to receive payouts",
  "individual.id_number": "Your SIN or national ID number",
  "individual.verification.document": "A photo of your government ID",
  "individual.verification.additional_document": "A second ID document",
  "individual.address.line1": "Your street address",
  "individual.dob.day": "Your date of birth",
  "company.tax_id": "The company's business number",
  "company.verification.document": "The company's incorporation document",
  "business_profile.url": "A website or public profile for the business",
  "business_profile.mcc": "The category the business operates in",
  "tos_acceptance.date": "Acceptance of Stripe's terms of service",
  "representative.verification.document": "ID for the account representative",
};

/**
 * A readable sentence when there is one, otherwise the raw Stripe name.
 *
 * The fallback is the name **unchanged** on purpose. Re-spacing it would
 * produce something that reads like a translation and carries no more
 * information than the token — and a reader could no longer tell which
 * requirements the product actually understands.
 */
export function readableRequirement(field: string): string {
  return REQUIREMENT_GLOSSARY[field] ?? field;
}

/** True when `field` has a real explanation rather than only its raw name. */
export function hasExplanation(field: string): boolean {
  return field in REQUIREMENT_GLOSSARY;
}

function RequirementList({
  title,
  fields,
  urgent,
}: {
  title: string;
  fields: string[];
  urgent?: boolean;
}) {
  if (fields.length === 0) return null;
  return (
    <div data-testid={urgent ? "requirements-past-due" : "requirements-currently-due"}>
      <p
        className={`text-xs font-medium ${urgent ? "text-accent-red" : "text-accent-gold"}`}
      >
        {title}
      </p>
      <ul className="mt-1 space-y-1">
        {fields.map((field) => (
          <li key={field} className="flex flex-wrap items-baseline gap-x-2 text-xs">
            <span className="text-text-primary">{readableRequirement(field)}</span>
            {/* The raw name, but only when it is not already what is shown —
                printing `external_account` twice helps nobody. */}
            {hasExplanation(field) && (
              <code className="font-mono text-[10px] text-text-muted">{field}</code>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

export function PayoutRequirements({ payouts }: { payouts?: ProviderPayoutState }) {
  // No object at all is the same epistemic state as `checked_live: false`: we
  // do not know. It must not fall through to a "ready" rendering.
  if (!payouts || !payouts.checked_live) {
    return (
      <div
        data-testid="payout-state-unknown"
        className="rounded-lg border border-border bg-surface/40 px-3 py-2.5"
      >
        <div className="flex items-start gap-2">
          <HelpCircle className="mt-0.5 h-4 w-4 shrink-0 text-text-muted" />
          <div>
            <p className="text-xs font-medium text-text-secondary">
              Payout status could not be checked
            </p>
            <p className="mt-0.5 text-[11px] leading-relaxed text-text-muted">
              We could not reach Stripe for this account just now, so this is not a
              statement that everything is in order. Refresh, or open the Stripe
              dashboard directly.
            </p>
          </div>
        </div>
      </div>
    );
  }

  const blocked = payouts.past_due.length > 0 || payouts.currently_due.length > 0;

  if (payouts.payouts_enabled && !blocked) {
    return (
      <div
        data-testid="payout-state-enabled"
        className="rounded-lg border border-emerald/25 bg-emerald/[0.06] px-3 py-2.5"
      >
        <div className="flex items-center gap-2">
          <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald" />
          <p className="text-xs font-medium text-emerald">
            Payouts enabled — Stripe has everything it needs
          </p>
        </div>
      </div>
    );
  }

  return (
    <div
      data-testid="payout-state-blocked"
      className="space-y-2 rounded-lg border border-accent-gold/25 bg-accent-gold/[0.06] px-3 py-2.5"
    >
      <div className="flex items-start gap-2">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-accent-gold" />
        <div className="min-w-0">
          <p className="text-xs font-medium text-accent-gold">
            {payouts.payouts_enabled
              ? "Payouts enabled, but Stripe still needs information"
              : "Payouts are not enabled yet"}
          </p>
          <p className="mt-0.5 text-[11px] leading-relaxed text-text-muted">
            {blocked
              ? "Stripe is waiting on the items below. Until they are supplied, completed jobs accrue earnings but cannot be paid out."
              : "Stripe has not enabled payouts on this account, and has not listed anything outstanding."}
          </p>
          {payouts.disabled_reason && (
            <p className="mt-1 text-[11px]">
              <span className="text-text-muted">Reason: </span>
              <code
                data-testid="payout-disabled-reason"
                className="font-mono text-[10px] text-text-secondary"
              >
                {payouts.disabled_reason}
              </code>
            </p>
          )}
        </div>
      </div>

      {/* Past due first: these are already overdue, and burying them under a
          longer "currently due" list is how the urgent one gets missed. */}
      <RequirementList title="Overdue" fields={payouts.past_due} urgent />
      <RequirementList title="Needed" fields={payouts.currently_due} />

      {!payouts.charges_enabled && (
        <p className="text-[11px] text-text-muted">
          Charges are also disabled on this account.
        </p>
      )}
    </div>
  );
}
