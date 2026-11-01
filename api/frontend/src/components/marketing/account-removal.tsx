"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { FormEvent, useEffect, useState } from "react";
import { ObfuscationSafeMailto } from "@/components/marketing/ObfuscationSafeMailto";
import { useAuth } from "@/lib/auth";
import { deleteAccount } from "@/lib/api";
import { useLocale } from "@/lib/locale";

type Copy = {
  kicker: string;
  title: string;
  whatTitle: string;
  what: string[];
  keepTitle: string;
  keep: string;
  facebookTitle: string;
  facebook: string[];
  statusTitle: string;
  statusIntro: string;
  codeLabel: string;
  check: string;
  missing: string;
  checking: string;
  working: string;
  done: string;
  noAccount: string;
  unavailable: string;
  deactivateTitle: string;
  deactivateBody: string;
  signedOut: string;
  signIn: string;
  confirmLabel: string;
  confirmPlaceholder: string;
  start: string;
  starting: string;
  account: string;
  otherPage: string;
  privacy: string;
};

const COPY: Record<"en" | "fr", Copy> = {
  en: {
    kicker: "Xcelsior Compute Inc. · Account and data deletion",
    title: "Delete your data",
    whatTitle: "What deletion removes",
    what: [
      "Your name, email address, password, and sign-in links for Facebook, Google, GitHub, and Hugging Face.",
      "Sessions, passkeys, API credentials, and the Facebook user id stored for sign-in.",
      "Profile details you saved in the dashboard.",
    ],
    keepTitle: "What deletion keeps",
    keep: "Billing records, invoices, and tax records stay for the period Canadian law requires, which is seven years. The deletion workflow records each store it clears and keeps a legal hold where the record cannot be erased yet.",
    facebookTitle: "Facebook",
    facebook: [
      "In Facebook, open Settings and privacy, then Apps and websites.",
      "Select Xcelsior and remove it. Removing the app detaches Facebook sign-in.",
      "On the Removed tab, open Xcelsior and choose to send a data-deletion request. Facebook then calls this site, and this page shows the result for the confirmation code.",
    ],
    statusTitle: "Check a deletion",
    statusIntro: "Enter the confirmation code from Facebook or from the deactivation page.",
    codeLabel: "Confirmation code",
    check: "Check status",
    missing: "No deletion is on file for that code.",
    checking: "Checking this confirmation code.",
    working: "Deletion is in progress. You can leave this page and come back with the same code.",
    done: "Deletion is complete, except for records kept for the legal period above.",
    noAccount: "Facebook asked us to delete data, and no Xcelsior account is linked to that Facebook user. There is nothing further to erase.",
    unavailable: "We received the request and could not record it. Email privacy@xcelsior.ca and include the confirmation code.",
    deactivateTitle: "Deactivate this account",
    deactivateBody: "Deactivation starts the same permanent deletion. You are signed out, and this page gives you a confirmation code you can check later.",
    signedOut: "Sign in to deactivate the account you are using.",
    signIn: "Sign in",
    confirmLabel: "Type DELETE to confirm",
    confirmPlaceholder: "DELETE",
    start: "Deactivate and delete",
    starting: "Starting deletion…",
    account: "Account",
    otherPage: "Deactivate a signed-in account",
    privacy: "Privacy policy",
  },
  fr: {
    kicker: "Xcelsior Compute Inc. · Suppression du compte et des données",
    title: "Supprimer vos données",
    whatTitle: "Ce que la suppression retire",
    what: [
      "Votre nom, votre adresse courriel, votre mot de passe et les liens de connexion Facebook, Google, GitHub et Hugging Face.",
      "Les sessions, les clés d'accès, les identifiants d'API et l'identifiant Facebook enregistré pour la connexion.",
      "Les renseignements de profil enregistrés dans le tableau de bord.",
    ],
    keepTitle: "Ce que la suppression conserve",
    keep: "Les dossiers de facturation, les factures et les dossiers fiscaux restent pour la période exigée par le droit canadien, soit sept ans. Le processus note chaque magasin de données qu'il vide et conserve une retenue lorsqu'un dossier ne peut pas encore être effacé.",
    facebookTitle: "Facebook",
    facebook: [
      "Dans Facebook, ouvrez Paramètres et confidentialité, puis Applications et sites Web.",
      "Choisissez Xcelsior et retirez-la. Retirer l'application détache la connexion Facebook.",
      "Dans l'onglet Retirées, ouvrez Xcelsior et envoyez une demande de suppression. Facebook appelle ensuite ce site, et cette page montre le résultat du code de confirmation.",
    ],
    statusTitle: "Vérifier une suppression",
    statusIntro: "Entrez le code de confirmation reçu de Facebook ou de la page de désactivation.",
    codeLabel: "Code de confirmation",
    check: "Vérifier",
    missing: "Aucune suppression n'est enregistrée pour ce code.",
    checking: "Vérification de ce code de confirmation.",
    working: "La suppression est en cours. Vous pouvez quitter cette page et revenir avec le même code.",
    done: "La suppression est terminée, sauf les dossiers conservés pour la période légale ci-dessus.",
    noAccount: "Facebook nous a demandé de supprimer des données, et aucun compte Xcelsior n'est lié à cet utilisateur Facebook. Il n'y a rien d'autre à effacer.",
    unavailable: "Nous avons reçu la demande et n'avons pas pu l'enregistrer. Écrivez à privacy@xcelsior.ca et joignez le code de confirmation.",
    deactivateTitle: "Désactiver ce compte",
    deactivateBody: "La désactivation lance la même suppression définitive. Vous êtes déconnecté, et cette page vous donne un code de confirmation à vérifier plus tard.",
    signedOut: "Connectez-vous pour désactiver le compte que vous utilisez.",
    signIn: "Connexion",
    confirmLabel: "Tapez DELETE pour confirmer",
    confirmPlaceholder: "DELETE",
    start: "Désactiver et supprimer",
    starting: "Démarrage de la suppression…",
    account: "Compte",
    otherPage: "Désactiver un compte connecté",
    privacy: "Politique de confidentialité",
  },
};

type Status = {
  outcome?: string;
  state?: string | null;
  deadline_at?: string | null;
  completed_at?: string | null;
  request_id?: string | null;
};

function statusMessage(copy: Copy, status: Status | null, missing: boolean): string {
  if (missing) return copy.missing;
  if (!status) return "";
  if (status.outcome === "no_account") return copy.noAccount;
  if (status.outcome === "unavailable") return copy.unavailable;
  if (status.state === "completed") return copy.done;
  if (status.outcome === "started" || status.state) return copy.working;
  return copy.missing;
}

export function DeleteDataContent() {
  const params = useSearchParams();
  const { displayLocale } = useLocale();
  const copy = COPY[displayLocale === "fr" ? "fr" : "en"];
  const initialCode = params.get("code") || "";
  const requestId = params.get("request") || "";
  const [code, setCode] = useState(initialCode);
  const [status, setStatus] = useState<Status | null>(null);
  const [missing, setMissing] = useState(false);
  const [checking, setChecking] = useState(false);

  useEffect(() => {
    if (!initialCode) return;
    let cancelled = false;
    setChecking(true);
    const query = new URLSearchParams({ code: initialCode });
    if (requestId) query.set("request", requestId);
    fetch(`/api/privacy/deletion-status?${query}`, { cache: "no-store" })
      .then(async (response) => {
        if (cancelled) return;
        if (response.status === 404) {
          setMissing(true);
          setStatus(null);
          return;
        }
        if (!response.ok) {
          setMissing(true);
          return;
        }
        setMissing(false);
        setStatus(await response.json());
      })
      .catch(() => {
        if (!cancelled) setMissing(true);
      })
      .finally(() => {
        if (!cancelled) setChecking(false);
      });
    return () => {
      cancelled = true;
    };
  }, [initialCode, requestId]);

  const onCheck = (event: FormEvent) => {
    event.preventDefault();
    const next = code.trim();
    if (!next) return;
    window.location.assign(`/delete-data?code=${encodeURIComponent(next)}`);
  };

  return (
    <article className="site-container">
      <div className="site-rails site-section site-legal-shell">
        <h1 className="site-section-heading site-legal-heading">{copy.title}</h1>
        <p className="site-legal-effective">{copy.kicker}</p>
        <div className="site-legal-body">
          <section className="site-legal-section">
            <h2 className="site-legal-title">{copy.whatTitle}</h2>
            <ul className="site-legal-list">
              {copy.what.map((item) => <li key={item}>{item}</li>)}
            </ul>
          </section>
          <section className="site-legal-section">
            <h2 className="site-legal-title">{copy.keepTitle}</h2>
            <p>{copy.keep}</p>
          </section>
          <section className="site-legal-section">
            <h2 className="site-legal-title">{copy.facebookTitle}</h2>
            <ul className="site-legal-list">
              {copy.facebook.map((item) => <li key={item}>{item}</li>)}
            </ul>
          </section>
          <section className="site-legal-section">
            <h2 className="site-legal-title">{copy.statusTitle}</h2>
            <p>{copy.statusIntro}</p>
            <form className="site-legal-form" onSubmit={onCheck}>
              <label className="site-legal-field" htmlFor="deletion-code">
                <span className="site-auth-label">{copy.codeLabel}</span>
                <input
                  id="deletion-code"
                  className="site-field"
                  value={code}
                  onChange={(event) => setCode(event.target.value)}
                  placeholder={copy.codeLabel}
                  autoComplete="off"
                  spellCheck={false}
                />
              </label>
              <button className="site-button site-button-primary" type="submit">{copy.check}</button>
            </form>
            {initialCode ? (
              <div className="site-legal-panel" aria-live="polite">
                <p>{checking ? copy.checking : statusMessage(copy, status, missing)}</p>
                {status?.request_id ? <p className="site-legal-code">{status.request_id}</p> : null}
                {status?.state ? <p className="site-legal-code">{status.state}</p> : null}
              </div>
            ) : null}
            <p className="site-legal-paragraph-gap">
              <Link href="/deactivate" className="site-inline-link">{copy.otherPage}</Link>
              {" · "}
              <Link href="/privacy" className="site-inline-link">{copy.privacy}</Link>
              {" · "}
              <ObfuscationSafeMailto href="mailto:privacy@xcelsior.ca" className="site-inline-link">
                privacy@xcelsior.ca
              </ObfuscationSafeMailto>
            </p>
          </section>
        </div>
      </div>
    </article>
  );
}

export function DeactivateContent() {
  const { displayLocale } = useLocale();
  const { user, loading } = useAuth();
  const copy = COPY[displayLocale === "fr" ? "fr" : "en"];
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState("");
  const [working, setWorking] = useState(false);

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (confirm !== "DELETE" || working) return;
    setWorking(true);
    setError("");
    try {
      const receipt = await deleteAccount(crypto.randomUUID());
      const query = new URLSearchParams({
        code: receipt.status_token,
        request: receipt.request_id,
      });
      window.location.assign(`/delete-data?${query}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : copy.unavailable);
      setWorking(false);
    }
  };

  return (
    <article className="site-container">
      <div className="site-rails site-section site-legal-shell">
        <h1 className="site-section-heading site-legal-heading">{copy.deactivateTitle}</h1>
        <p className="site-legal-effective">{copy.kicker}</p>
        <div className="site-legal-body">
          <section className="site-legal-section">
            <h2 className="site-legal-title">{copy.whatTitle}</h2>
            <p>{copy.deactivateBody}</p>
            <ul className="site-legal-list site-legal-list-gap">
              {copy.what.map((item) => <li key={item}>{item}</li>)}
            </ul>
          </section>
          <section className="site-legal-section">
            <h2 className="site-legal-title">{copy.keepTitle}</h2>
            <p>{copy.keep}</p>
          </section>
          <section className="site-legal-section">
            <h2 className="site-legal-title">{copy.deactivateTitle}</h2>
            {loading ? <p>{copy.working}</p> : null}
            {!loading && !user ? (
              <div className="site-legal-panel">
                <p>{copy.signedOut}</p>
                <div className="site-legal-actions">
                  <Link href="/login?redirect=%2Fdeactivate" className="site-button site-button-primary">{copy.signIn}</Link>
                </div>
              </div>
            ) : null}
            {!loading && user ? (
              <form className="site-legal-panel" onSubmit={onSubmit}>
                <p>{copy.account}: {user.email}</p>
                <div className="site-legal-form">
                  <label className="site-legal-field" htmlFor="deactivate-confirm">
                    <span className="site-auth-label">{copy.confirmLabel}</span>
                    <input
                      id="deactivate-confirm"
                      className="site-field"
                      value={confirm}
                      onChange={(event) => setConfirm(event.target.value)}
                      placeholder={copy.confirmPlaceholder}
                      autoComplete="off"
                      spellCheck={false}
                    />
                  </label>
                  <button
                    className="site-button site-button-primary"
                    type="submit"
                    disabled={confirm !== "DELETE" || working}
                  >
                    {working ? copy.starting : copy.start}
                  </button>
                </div>
                {error ? <p>{error}</p> : null}
              </form>
            ) : null}
          </section>
          <section className="site-legal-section">
            <h2 className="site-legal-title">{copy.facebookTitle}</h2>
            <ul className="site-legal-list">
              {copy.facebook.map((item) => <li key={item}>{item}</li>)}
            </ul>
            <p className="site-legal-paragraph-gap">
              <Link href="/delete-data" className="site-inline-link">{copy.title}</Link>
              {" · "}
              <Link href="/privacy" className="site-inline-link">{copy.privacy}</Link>
            </p>
          </section>
        </div>
      </div>
    </article>
  );
}
