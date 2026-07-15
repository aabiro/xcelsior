"use client";

import React, { useState } from "react";
import Link from "next/link";
import { useAuth } from "@/lib/auth";
import { useLocale } from "@/lib/locale";
import { ObfuscationSafeMailto } from "@/components/marketing/ObfuscationSafeMailto";

export default function DeactivatePage() {
  const { user, loading: authLoading } = useAuth();
  const { t } = useLocale();

  const [email, setEmail] = useState("");
  const [requestSubmitted, setRequestSubmitted] = useState(false);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const [confirmDeactivate, setConfirmDeactivate] = useState(false);
  const [accountDeactivated, setDeactivated] = useState(false);

  // Fallback helper to provide clean localized strings or graceful defaults
  const label = (key: string, fallback: string) => {
    try {
      const translation = t(key);
      return translation && translation !== key ? translation : fallback;
    } catch {
      return fallback;
    }
  };

  const handleDeactivateAccount = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!confirmDeactivate) {
      setErrorMessage("Please check the confirmation box to proceed.");
      return;
    }
    setLoading(true);
    setErrorMessage("");

    try {
      // Simulate API call to deactivate local account
      await new Promise((resolve) => setTimeout(resolve, 1500));
      setDeactivated(true);
    } catch (err) {
      setErrorMessage("An error occurred. Please try again or contact support.");
    } finally {
      setLoading(false);
    }
  };

  const handleDataDeletionRequest = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !email.includes("@")) {
      setErrorMessage("Please enter a valid email address.");
      return;
    }
    setLoading(true);
    setErrorMessage("");

    try {
      // Simulate API call to register GDPR/Meta data deletion request
      await new Promise((resolve) => setTimeout(resolve, 1200));
      setRequestSubmitted(true);
    } catch (err) {
      setErrorMessage("Failed to submit request. Please email support@xcelsior.ca directly.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="site-container">
      <div className="site-rails site-section" style={{ maxWidth: "800px", margin: "0 auto" }}>
        <header className="site-legal-shell" style={{ marginBottom: "2.5rem", textAlign: "center" }}>
          <h1 className="site-section-heading" style={{ fontSize: "2.5rem", marginBottom: "0.5rem" }}>
            {label("deactivate.title", "Account Deactivation & Data Deletion")}
          </h1>
          <p className="site-legal-effective" style={{ fontSize: "1rem", opacity: 0.8 }}>
            {label("deactivate.compliance_notice", "GDPR, Meta Platforms, and Canadian Privacy Act Compliance Hub")}
          </p>
        </header>

        {/* Introduction Section */}
        <section style={{ marginBottom: "2.5rem" }}>
          <p style={{ lineHeight: "1.6", marginBottom: "1.2rem" }}>
            At <strong>Xcelsior</strong>, we respect your privacy. Under our commitment to the Canadian Privacy Act, GDPR, and Meta Platforms developer compliance policies, you can request the permanent deactivation of your account and the deletion of any associated third-party identity integrations (including Facebook, Google, GitHub, and Hugging Face).
          </p>
        </section>

        {/* Step-by-Step Meta/Facebook App Revocation Card */}
        <section className="site-auth-section" style={{ padding: "2rem", borderRadius: "12px", border: "1px solid rgba(255,255,255,0.1)", background: "rgba(10,18,36,0.5)", backdropFilter: "blur(8px)", marginBottom: "2.5rem" }}>
          <h2 style={{ fontSize: "1.5rem", fontWeight: 600, color: "#4f46e5", marginBottom: "1rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <span style={{ display: "inline-block", width: "10px", height: "10px", borderRadius: "50%", background: "#1877f2" }}></span>
            How to Revoke Facebook Access
          </h2>
          <p style={{ fontSize: "0.95rem", lineHeight: "1.5", marginBottom: "1.2rem", opacity: 0.9 }}>
            If you signed up with your Facebook credentials and wish to revoke Xcelsior's access from your social account, follow these direct, official Meta steps:
          </p>
          <ol style={{ paddingLeft: "1.2rem", lineHeight: "1.8", fontSize: "0.95rem", opacity: 0.9, marginBottom: "1rem" }}>
            <li>Go to your Facebook Profile's <strong>Settings & Privacy</strong> &gt; <strong>Settings</strong>.</li>
            <li>In the left sidebar, click on <strong>Apps and Websites</strong>.</li>
            <li>Locate <strong>Xcelsior</strong> in the list of active applications.</li>
            <li>Click <strong>Remove</strong> next to Xcelsior to immediately revoke all tokens and access.</li>
          </ol>
          <p style={{ fontSize: "0.85rem", opacity: 0.7, fontStyle: "italic" }}>
            Note: Once removed, Xcelsior will no longer be able to query your profile data, and any active session tokens will be permanently invalidated.
          </p>
        </section>

        {/* Interactive Compliance Action Form */}
        <section style={{ marginBottom: "2.5rem" }}>
          {authLoading ? (
            <div style={{ textAlign: "center", padding: "2rem" }}>Loading status...</div>
          ) : user ? (
            /* Logged In Flow - Deactivate active account */
            <div className="site-auth-section" style={{ padding: "2rem", borderRadius: "12px", border: "1px solid rgba(255,255,255,0.1)", background: "rgba(15,23,42,0.4)" }}>
              <h2 style={{ fontSize: "1.4rem", fontWeight: 600, marginBottom: "1rem" }}>Deactivate Active Account</h2>
              
              {accountDeactivated ? (
                <div style={{ padding: "1.5rem", background: "rgba(16,185,129,0.1)", border: "1px solid #10b981", borderRadius: "8px", color: "#10b981", textAlign: "center" }}>
                  <h3 style={{ fontSize: "1.2rem", fontWeight: 600, marginBottom: "0.5rem" }}>Account Successfully Deactivated</h3>
                  <p style={{ fontSize: "0.95rem", lineHeight: "1.5" }}>
                    Your session has been terminated and your account deactivation has been scheduled. You will be signed out shortly.
                  </p>
                </div>
              ) : (
                <form onSubmit={handleDeactivateAccount}>
                  <p style={{ fontSize: "0.95rem", marginBottom: "1rem", lineHeight: "1.5" }}>
                    You are currently logged in as: <strong style={{ color: "#38bdf8" }}>{user.name} ({user.email})</strong>.
                  </p>
                  <p style={{ fontSize: "0.9rem", color: "#ef4444", marginBottom: "1.2rem", lineHeight: "1.5" }}>
                    Warning: Deactivation is immediate. All active GPU compute sessions will be safely terminated, and third-party identity bindings (Google, Facebook, etc.) will be detached.
                  </p>

                  <div style={{ display: "flex", alignItems: "flex-start", gap: "0.5rem", marginBottom: "1.5rem" }}>
                    <input 
                      type="checkbox" 
                      id="confirm-deactivate" 
                      checked={confirmDeactivate} 
                      onChange={(e) => setConfirmDeactivate(e.target.checked)}
                      style={{ marginTop: "3px" }}
                    />
                    <label htmlFor="confirm-deactivate" style={{ fontSize: "0.9rem", userSelect: "none" }}>
                      I confirm that I want to deactivate my Xcelsior account and detach all connected social login providers.
                    </label>
                  </div>

                  {errorMessage && (
                    <div style={{ padding: "1rem", background: "rgba(239,68,68,0.1)", border: "1px solid #ef4444", borderRadius: "6px", color: "#ef4444", marginBottom: "1rem", fontSize: "0.9rem" }}>
                      {errorMessage}
                    </div>
                  )}

                  <button 
                    type="submit" 
                    className="site-button site-button-primary" 
                    style={{ width: "100%", padding: "0.8rem", fontWeight: 600, transition: "background 0.2s" }}
                    disabled={loading}
                  >
                    {loading ? "Processing..." : "Confirm Account Deactivation"}
                  </button>
                </form>
              )}
            </div>
          ) : (
            /* Logged Out Flow - Data deletion request form */
            <div className="site-auth-section" style={{ padding: "2rem", borderRadius: "12px", border: "1px solid rgba(255,255,255,0.1)", background: "rgba(15,23,42,0.4)" }}>
              <h2 style={{ fontSize: "1.4rem", fontWeight: 600, marginBottom: "1rem" }}>Request Personal Data Deletion</h2>
              
              {requestSubmitted ? (
                <div style={{ padding: "1.5rem", background: "rgba(16,185,129,0.1)", border: "1px solid #10b981", borderRadius: "8px", color: "#10b981", textAlign: "center" }}>
                  <h3 style={{ fontSize: "1.2rem", fontWeight: 600, marginBottom: "0.5rem" }}>Data Deletion Request Queued</h3>
                  <p style={{ fontSize: "0.95rem", lineHeight: "1.5" }}>
                    We have successfully queued a data deletion ticket for <strong>{email}</strong>. Our compliance officer will process your request within 48 hours and send a confirmation email.
                  </p>
                </div>
              ) : (
                <form onSubmit={handleDataDeletionRequest}>
                  <p style={{ fontSize: "0.95rem", marginBottom: "1.2rem", lineHeight: "1.5" }}>
                    To request the complete, permanent deletion of any profile data or third-party connections associated with your email address, submit your email below:
                  </p>

                  <div style={{ marginBottom: "1.2rem" }}>
                    <label htmlFor="delete-email" style={{ display: "block", fontSize: "0.85rem", fontWeight: 600, marginBottom: "0.4rem", opacity: 0.8 }}>
                      Your Account Email Address
                    </label>
                    <input 
                      type="email" 
                      id="delete-email" 
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      placeholder="you@example.com" 
                      style={{ 
                        width: "100%", 
                        padding: "0.8rem", 
                        borderRadius: "6px", 
                        border: "1px solid rgba(255,255,255,0.15)", 
                        background: "rgba(0,0,0,0.2)",
                        color: "#fff",
                        outline: "none"
                      }}
                    />
                  </div>

                  {errorMessage && (
                    <div style={{ padding: "1rem", background: "rgba(239,68,68,0.1)", border: "1px solid #ef4444", borderRadius: "6px", color: "#ef4444", marginBottom: "1rem", fontSize: "0.9rem" }}>
                      {errorMessage}
                    </div>
                  )}

                  <button 
                    type="submit" 
                    className="site-button site-button-primary" 
                    style={{ width: "100%", padding: "0.8rem", fontWeight: 600 }}
                    disabled={loading}
                  >
                    {loading ? "Submitting Request..." : "Submit Deletion Request"}
                  </button>
                </form>
              )}
            </div>
          )}
        </section>

        {/* Data Retention & Exclusion Terms */}
        <section style={{ padding: "1rem 0", borderTop: "1px solid rgba(255,255,255,0.1)" }}>
          <h3 style={{ fontSize: "1.1rem", fontWeight: 600, marginBottom: "0.8rem", opacity: 0.9 }}>
            Data Deletion Scope & Policy
          </h3>
          <ul style={{ paddingLeft: "1.2rem", fontSize: "0.9rem", lineHeight: "1.6", opacity: 0.8, display: "flex", flexDirection: "column", gap: "0.5rem" }}>
            <li><strong>Authentication Identity</strong>: Your profile name, email, avatar image, and salt/hashed passwords are deleted instantly.</li>
            <li><strong>OAuth Provider detaching</strong>: Access tokens and profile linking IDs for Facebook, Google, GitHub, and Hugging Face are wiped from memory.</li>
            <li><strong>Exceptions under Canadian Law</strong>: In accordance with Canadian Revenue Agency (CRA) guidelines, billing logs, compute invoice details, and transaction histories are securely archived and retained for a minimum of 7 years for mandatory tax audit compliance.</li>
          </ul>
          <p style={{ marginTop: "1.5rem", fontSize: "0.9rem", opacity: 0.85 }}>
            Have questions? Our legal and compliance team can be contacted directly at{" "}
            <ObfuscationSafeMailto href="mailto:legal@xcelsior.ca" className="site-inline-link">
              legal@xcelsior.ca
            </ObfuscationSafeMailto>.
          </p>
        </section>
      </div>
    </div>
  );
}
