"use client";

import { useCallback, useMemo, useState } from "react";
import Image from "next/image";
import Link from "next/link";
import {
  Bot, Copy, CheckCircle, Loader2, ChevronRight, ExternalLink, KeyRound,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input, Label } from "@/components/ui/input";
import { SettingsSection } from "@/components/settings/settings-layout";
import { OAuthSecretRevealModal } from "@/components/settings/oauth-secret-reveal-modal";
import { ScopeChipRow } from "@/components/settings/credential-scope-panel";
import { CodeBlock } from "@/components/ui/code-block";
import { useLocale } from "@/lib/locale";
import * as api from "@/lib/api";
import type { OAuthClientInfo } from "@/lib/api";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

const MCP_SCOPES = [
  "instances:read",
  "instances:write",
  "billing:read",
  "gpu:read",
  "marketplace:read",
] as const;

const STEPS = ["create", "copy", "token", "paste"] as const;

const AGENTS = [
  { id: "cursor", art: "/mcp/agent-cursor.svg", labelKey: "dash.settings.mcp.agent_cursor" },
  { id: "claude", art: "/mcp/agent-claude.svg", labelKey: "dash.settings.mcp.agent_claude" },
  { id: "vscode", art: "/mcp/agent-vscode.svg", labelKey: "dash.settings.mcp.agent_vscode" },
  { id: "github", art: "/logos/github.svg", labelKey: "dash.settings.mcp.agent_github" },
] as const;

type FreshClientCredentials = {
  clientId: string;
  clientSecret: string;
  name: string;
};

function mcpUrl(): string {
  if (typeof window !== "undefined" && window.location.hostname === "localhost") {
    return "http://localhost:8770/mcp";
  }
  return "https://xcelsior.ca/mcp";
}

// MCP client config formats differ by agent:
//  - Cursor  (~/.cursor/mcp.json):  `mcpServers` + `url` (transport inferred)
//  - Claude  (.mcp.json):           `mcpServers` + explicit `type: "http"`
//  - VS Code (.vscode/mcp.json):    `servers` (not mcpServers) + `type: "http"`
function configJson(agentId: string, tokenPlaceholder = "YOUR_OAUTH_TOKEN"): string {
  const url = mcpUrl();
  const headers = { Authorization: `Bearer ${tokenPlaceholder}` };
  if (agentId === "github") {
    return JSON.stringify(
      {
        mcpServers: {
          "xcelsior-readonly": {
            type: "http",
            url,
            headers: {
              Authorization: "Bearer ${COPILOT_MCP_XCELSIOR_ACCESS_TOKEN}",
            },
            tools: [
              "list_available_gpus",
              "get_spot_prices",
              "get_pricing_reference",
              "search_marketplace",
              "list_tiers",
            ],
          },
        },
      },
      null,
      2,
    );
  }
  if (agentId === "vscode") {
    return JSON.stringify({ servers: { xcelsior: { type: "http", url, headers } } }, null, 2);
  }
  if (agentId === "claude") {
    return JSON.stringify({ mcpServers: { xcelsior: { type: "http", url, headers } } }, null, 2);
  }
  return JSON.stringify({ mcpServers: { xcelsior: { url, headers } } }, null, 2);
}

// Where each client expects the config file to live (shown above the snippet).
function configPath(agentId: string): string {
  if (agentId === "github") return "GitHub -> Settings -> Copilot -> MCP servers";
  if (agentId === "vscode") return ".vscode/mcp.json";
  if (agentId === "claude") return ".mcp.json (project root) or claude_desktop_config.json";
  return "~/.cursor/mcp.json";
}

function tokenCurl(clientId: string, clientSecret: string): string {
  const base = process.env.NEXT_PUBLIC_API_URL || "https://xcelsior.ca";
  return `curl -s -X POST '${base}/oauth/token' \\
  -H 'Content-Type: application/x-www-form-urlencoded' \\
  -d 'grant_type=client_credentials&client_id=${clientId}&client_secret=${clientSecret}'`;
}

export function McpAgentSetup({
  oauthClients,
  onOAuthClientsChange,
}: {
  oauthClients: OAuthClientInfo[];
  onOAuthClientsChange: (clients: OAuthClientInfo[]) => void;
}) {
  const { t } = useLocale();
  const [step, setStep] = useState(0);
  const [creating, setCreating] = useState(false);
  const [agent, setAgent] = useState<(typeof AGENTS)[number]["id"]>("cursor");
  const [freshCredentials, setFreshCredentials] = useState<FreshClientCredentials | null>(null);
  const [showSecretModal, setShowSecretModal] = useState(false);
  const [copied, setCopied] = useState<string | null>(null);
  const [testing, setTesting] = useState(false);
  const [minting, setMinting] = useState(false);
  // The live Bearer token: minted in-app or pasted by hand. When set, it's spliced
  // straight into the config preview so there's no YOUR_OAUTH_TOKEN left to edit.
  const [token, setToken] = useState("");

  const mcpClient = useMemo(
    () => oauthClients.find((c) => c.client_name?.toLowerCase().includes("mcp") || c.scopes?.includes("gpu:read")),
    [oauthClients],
  );

  const copyText = useCallback((key: string, text: string) => {
    void navigator.clipboard.writeText(text);
    setCopied(key);
    toast.success(t("dash.settings.mcp.copied"));
    setTimeout(() => setCopied(null), 2000);
  }, [t]);

  const handleCreateClient = async () => {
    setCreating(true);
    try {
      const name = `mcp-agent-${new Date().toISOString().slice(0, 10)}`;
      const res = await api.createOAuthClient(name, [...MCP_SCOPES]);
      const client = res.client;
      const { client_secret: _secret, ...listed } = client;
      onOAuthClientsChange([...oauthClients, listed]);
      if (client.client_secret) {
        setFreshCredentials({
          clientId: client.client_id,
          clientSecret: client.client_secret,
          name: client.client_name,
        });
        setShowSecretModal(true);
      }
      setStep(1);
      toast.success(t("dash.settings.mcp.client_created"));
    } catch {
      toast.error(t("dash.settings.mcp.client_failed"));
    } finally {
      setCreating(false);
    }
  };

  // Exchange the freshly-revealed client credentials for a Bearer token directly,
  // so the user never has to leave the page to run the curl by hand. The secret is
  // only available for this setup session (right after creation / rotation).
  const handleGetToken = async () => {
    if (!freshCredentials) return;
    setMinting(true);
    try {
      const base = process.env.NEXT_PUBLIC_API_URL || "";
      const res = await fetch(`${base}/oauth/token`, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({
          grant_type: "client_credentials",
          client_id: freshCredentials.clientId,
          client_secret: freshCredentials.clientSecret,
        }).toString(),
      });
      const body = (await res.json()) as { access_token?: string };
      if (res.ok && body.access_token) {
        setToken(body.access_token);
        setStep(3);
        toast.success(t("dash.settings.mcp.token_minted"));
      } else {
        toast.error(t("dash.settings.mcp.token_failed"));
      }
    } catch {
      toast.error(t("dash.settings.mcp.token_failed"));
    } finally {
      setMinting(false);
    }
  };

  const handleTestHealth = async () => {
    const bearer = token.trim();
    if (!bearer) {
      toast.error(t("dash.settings.mcp.token_required"));
      setStep(Math.max(step, 2));
      return;
    }
    setTesting(true);
    try {
      // MCP health lives at /mcp/health (nginx proxies it to the MCP service in
      // prod; the server also matches /mcp/health directly in local dev).
      // The old code stripped /mcp and hit the API root /health by mistake.
      const url = `${mcpUrl()}/health`;
      const res = await fetch(url, { headers: { Authorization: `Bearer ${bearer}` } });
      const body = (await res.json()) as { status?: string };
      const base = process.env.NEXT_PUBLIC_API_URL || "";
      const tokenRes = await fetch(`${base}/api/auth/introspect`, {
        headers: { Authorization: `Bearer ${bearer}` },
      });
      if (res.ok && body.status === "healthy" && tokenRes.ok) {
        setStep(3);
        toast.success(t("dash.settings.mcp.test_ok"));
      } else {
        toast.error(t("dash.settings.mcp.test_fail"));
      }
    } catch {
      toast.error(t("dash.settings.mcp.test_fail"));
    } finally {
      setTesting(false);
    }
  };

  return (
    <SettingsSection
      title={t("dash.settings.mcp.title")}
      description={t("dash.settings.mcp.subtitle")}
      icon={Bot}
      accent="violet"
    >
      <div className="mb-6 overflow-hidden rounded-xl border border-border/60">
        <Image
          src="/mcp/settings-hero.svg"
          alt=""
          width={480}
          height={160}
          className="h-auto w-full"
        />
      </div>

      <div className="grid gap-8 lg:grid-cols-[1fr_1fr]">
        <div className="space-y-4">
          <div className="flex gap-2">
            {STEPS.map((s, i) => (
              <div
                key={s}
                className={cn(
                  "flex h-8 w-8 items-center justify-center rounded-full text-xs font-bold",
                  i <= step ? "bg-accent-violet text-white" : "bg-border/40 text-text-muted",
                )}
              >
                {i < step ? <CheckCircle className="h-4 w-4" /> : i + 1}
              </div>
            ))}
          </div>

          {step === 0 && (
            <div className="space-y-3">
              <p className="text-sm text-text-secondary">{t("dash.settings.mcp.step_create")}</p>
              <ScopeChipRow scopes={[...MCP_SCOPES]} maxVisible={20} size="md" />
              <Button onClick={handleCreateClient} disabled={creating} className="gap-2">
                {creating ? <Loader2 className="h-4 w-4 animate-spin" /> : <Bot className="h-4 w-4" />}
                {t("dash.settings.mcp.create_btn")}
              </Button>
              {mcpClient && (
                <p className="text-xs text-text-muted">
                  {t("dash.settings.mcp.existing_client")}: <code>{mcpClient.client_name}</code>
                  <button type="button" className="ml-2 text-accent-cyan underline" onClick={() => setStep(1)}>
                    {t("dash.settings.mcp.use_existing")}
                  </button>
                </p>
              )}
            </div>
          )}

          {step >= 1 && mcpClient && (
            <div className="space-y-3 rounded-xl border border-border/60 bg-surface/30 p-4">
              <p className="text-sm font-medium">{t("dash.settings.mcp.step_copy")}</p>
              <div>
                <Label className="text-xs">{t("dash.settings.oauth.client_id_label")}</Label>
                <div className="mt-1 flex gap-2">
                  <Input readOnly value={mcpClient.client_id} className="font-mono text-xs" />
                  <Button size="icon" variant="outline" onClick={() => copyText("id", mcpClient.client_id)}>
                    {copied === "id" ? <CheckCircle className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
              </div>
              <p className="text-xs text-text-muted">{t("dash.settings.mcp.secret_hint")}</p>
              <Button variant="outline" size="sm" onClick={() => setStep(2)}>
                {t("dash.settings.mcp.next_token")} <ChevronRight className="ml-1 h-3 w-3" />
              </Button>
            </div>
          )}

          {step >= 2 && mcpClient && (
            <div className="space-y-3 rounded-xl border border-border/60 bg-surface/30 p-4">
              <p className="text-sm font-medium">{t("dash.settings.mcp.step_token")}</p>
              {freshCredentials ? (
                <Button size="sm" onClick={handleGetToken} disabled={minting} className="gap-2">
                  {minting ? <Loader2 className="h-4 w-4 animate-spin" /> : <KeyRound className="h-4 w-4" />}
                  {minting ? t("dash.settings.mcp.minting") : t("dash.settings.mcp.get_token_btn")}
                </Button>
              ) : (
                <p className="text-xs text-text-muted">{t("dash.settings.mcp.token_secret_missing")}</p>
              )}
              {token && (
                <div>
                  <Label className="text-xs">{t("dash.settings.mcp.minted_token_label")}</Label>
                  <div className="mt-1 flex gap-2">
                    <Input readOnly value={token} className="font-mono text-xs" />
                    <Button size="icon" variant="outline" onClick={() => copyText("token", token)}>
                      {copied === "token" ? <CheckCircle className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                    </Button>
                  </div>
                </div>
              )}
              {freshCredentials && (
                <details className="text-xs text-text-muted">
                  <summary className="cursor-pointer select-none hover:text-text-secondary">
                    {t("dash.settings.mcp.copy_curl")}
                  </summary>
                  <div className="mt-2">
                    <CodeBlock
                      filename="get-token.sh"
                      code={tokenCurl(freshCredentials.clientId, freshCredentials.clientSecret)}
                      onCopy={undefined}
                    />
                  </div>
                </details>
              )}
              <Button variant="outline" size="sm" onClick={() => setStep(3)}>
                {t("dash.settings.mcp.next_paste")} <ChevronRight className="ml-1 h-3 w-3" />
              </Button>
            </div>
          )}

          {step >= 3 && (
            <div className="space-y-2">
              <p className="text-sm font-medium">{t("dash.settings.mcp.step_paste")}</p>
              <div className="flex gap-2">
                {AGENTS.map((a) => (
                  <button
                    key={a.id}
                    type="button"
                    onClick={() => setAgent(a.id)}
                    className={cn(
                      "flex items-center gap-1.5 rounded-lg border px-2 py-1.5 text-xs",
                      agent === a.id ? "border-accent-violet/50 bg-accent-violet/10" : "border-border/60",
                    )}
                  >
                    <Image src={a.art} alt="" width={16} height={16} />
                    {t(a.labelKey)}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="space-y-3">
          <p className="text-xs font-medium uppercase tracking-wider text-text-muted">
            {t("dash.settings.mcp.preview")}
          </p>
          <CodeBlock
            filename={configPath(agent)}
            code={configJson(agent, token || undefined)}
            onCopy={undefined}
          />
          <div>
            <Label className="text-xs">{t("dash.settings.mcp.token_input_label")}</Label>
            <Input
              value={token}
              onChange={(e) => setToken(e.target.value)}
              placeholder={t("dash.settings.mcp.token_input_placeholder")}
              className="mt-1 font-mono text-xs"
            />
            <p className="mt-1 text-[11px] text-text-muted">{t("dash.settings.mcp.token_input_hint")}</p>
            {agent === "github" && token && (
              <p className="mt-1 text-[11px] text-accent-cyan">
                {t("dash.settings.mcp.github_secret_hint")}
              </p>
            )}
          </div>
          <div className="flex flex-wrap gap-2">
            <Button size="sm" variant="outline" onClick={handleTestHealth} disabled={testing}>
              {testing ? <Loader2 className="mr-1 h-3 w-3 animate-spin" /> : null}
              {t("dash.settings.mcp.test_health")}
            </Button>
            <Link href="/mcp" className="inline-flex items-center gap-1 text-sm text-text-secondary hover:text-text-primary">
              {t("dash.settings.mcp.docs_link")} <ExternalLink className="h-3 w-3" />
            </Link>
          </div>
        </div>
      </div>

      {freshCredentials && (
        <OAuthSecretRevealModal
          open={showSecretModal}
          onClose={() => setShowSecretModal(false)}
          clientId={freshCredentials.clientId}
          clientSecret={freshCredentials.clientSecret}
          scopes={[...MCP_SCOPES]}
        />
      )}
    </SettingsSection>
  );
}
