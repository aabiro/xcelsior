"use client";

import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Play, Square, Loader2, Terminal, MessageSquare, Code2, Copy, Check,
} from "lucide-react";
import type { ServerlessEndpoint } from "@/lib/api";
import * as api from "@/lib/api";
import { useLocale } from "@/lib/locale";
import { toast } from "sonner";
import posthog from "posthog-js";

import { CopyableText } from "./copyable-text";
import { ServerlessPanel, ServerlessSegmentedTabs } from "./serverless-ui";

type ConsoleMode = "chat" | "job" | "snippets";

/** Classify a preset model so the snippets show the right OpenAI route. Mirrors
 *  serverless/openai_proxy.py:model_task. */
function presetTaskFor(modelId: string): "chat" | "embed" | "rerank" {
  const s = modelId.toLowerCase();
  if (/rerank|cross-encoder/.test(s)) return "rerank";
  if (/bge-m3|bge-large|bge-base|bge-small|nomic-embed|gte-|e5-(?:large|base|small)|stella|snowflake-arctic-embed|embed/.test(s)) {
    return "embed";
  }
  return "chat";
}

interface TryItConsoleProps {
  endpoint: ServerlessEndpoint;
  canWrite: boolean;
}

export function TryItConsole({ endpoint, canWrite }: TryItConsoleProps) {
  const { t } = useLocale();
  const [mode, setMode] = useState<ConsoleMode>("chat");
  const [prompt, setPrompt] = useState("Hello! Summarize what you can do in one sentence.");
  const [jobPayload, setJobPayload] = useState('{"message": "hello"}');
  const [output, setOutput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const esRef = useRef<EventSource | null>(null);
  const outputRef = useRef<HTMLPreElement>(null);

  const endpointId = endpoint.endpoint_id;
  const isPreset = endpoint.mode === "preset";
  const modelId = endpoint.model_ref || endpoint.model_name || endpoint.model_id || "model";
  const baseUrl = typeof window !== "undefined"
    ? `${window.location.origin}${endpoint.openai_base_url || `/v1/serverless/${endpointId}/openai/v1`}`
    : endpoint.openai_base_url || "";

  useEffect(() => {
    return () => {
      esRef.current?.close();
    };
  }, []);

  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [output]);

  const stopStream = () => {
    esRef.current?.close();
    esRef.current = null;
    setStreaming(false);
    if (activeJobId && canWrite) {
      api.cancelServerlessJob(endpointId, activeJobId).catch(() => {});
    }
    setActiveJobId(null);
  };

  const runChat = async (stream: boolean) => {
    if (!canWrite) return toast.error(t("dash.serverless.viewer_blocked"));
    if (!isPreset) return toast.error(t("dash.serverless.chat_preset_only"));
    setOutput("");
    setStreaming(true);
    try {
      const text = await api.serverlessOpenAIChat(
        endpointId,
        {
          model: modelId,
          messages: [{ role: "user", content: prompt }],
          max_tokens: 512,
          stream,
        },
        stream ? (chunk) => setOutput((prev) => prev + chunk) : undefined,
      );
      if (!stream) setOutput(text);
      // Activation funnel: the user actually ran an inference.
      posthog.capture("serverless_inference_run", {
        mode: "chat",
        model: modelId,
        task: presetTaskFor(modelId),
        stream,
      });
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : t("dash.serverless.run_failed"));
    } finally {
      setStreaming(false);
    }
  };

  const runJob = async () => {
    if (!canWrite) return toast.error(t("dash.serverless.viewer_blocked"));
    let input: Record<string, unknown>;
    try {
      input = JSON.parse(jobPayload);
    } catch {
      return toast.error(t("dash.serverless.invalid_json"));
    }
    setOutput("");
    setStreaming(true);
    try {
      const res = await api.runServerlessJob(endpointId, input);
      setActiveJobId(res.id);
      setOutput(`Job ${res.id} — ${res.status}\n`);
      // Activation funnel: the user actually ran an inference (custom job).
      posthog.capture("serverless_inference_run", { mode: "job", model: modelId });

      const es = api.createServerlessJobStream(endpointId, res.id);
      esRef.current = es;
      es.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data);
          const chunk = data?.payload?.text ?? data?.payload?.output ?? data?.text ?? "";
          if (chunk) setOutput((prev) => prev + (typeof chunk === "string" ? chunk : JSON.stringify(chunk)));
          if (data?.event_type === "done" || data?.event_type === "error") {
            stopStream();
          }
        } catch {
          if (e.data && e.data !== "[DONE]") setOutput((prev) => prev + e.data + "\n");
        }
      };
      es.onerror = () => {
        api.getServerlessJobStatus(endpointId, res.id).then((st) => {
          if (st.output) setOutput((prev) => prev + JSON.stringify(st.output, null, 2));
          if (st.error) setOutput((prev) => prev + JSON.stringify(st.error, null, 2));
        }).finally(stopStream);
      };
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : t("dash.serverless.run_failed"));
      setStreaming(false);
    }
  };

  const task = isPreset ? presetTaskFor(modelId) : "chat";

  const presetCurl =
    task === "embed"
      ? `curl -X POST '${baseUrl}/embeddings' \\
  -H 'Content-Type: application/json' \\
  -H 'Authorization: Bearer YOUR_API_KEY' \\
  -d '{
    "model": "${modelId}",
    "input": ["The quick brown fox jumps over the lazy dog"]
  }'`
      : task === "rerank"
      ? `curl -X POST '${baseUrl}/rerank' \\
  -H 'Content-Type: application/json' \\
  -H 'Authorization: Bearer YOUR_API_KEY' \\
  -d '{
    "model": "${modelId}",
    "query": "What is the capital of France?",
    "documents": ["Paris is the capital of France.", "Berlin is in Germany."]
  }'`
      : `curl -X POST '${baseUrl}/chat/completions' \\
  -H 'Content-Type: application/json' \\
  -H 'Authorization: Bearer YOUR_API_KEY' \\
  -d '{
    "model": "${modelId}",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'`;

  const curlSnippet = isPreset
    ? presetCurl
    : `curl -X POST '${typeof window !== "undefined" ? window.location.origin : ""}/v1/serverless/${endpointId}/run' \\
  -H 'Content-Type: application/json' \\
  -H 'Authorization: Bearer YOUR_API_KEY' \\
  -d '{"input": ${jobPayload}}'`;

  const openaiSnippet =
    task === "embed"
      ? `from openai import OpenAI

client = OpenAI(base_url="${baseUrl}", api_key="YOUR_API_KEY")

resp = client.embeddings.create(
    model="${modelId}",
    input=["The quick brown fox jumps over the lazy dog"],
)
print(resp.data[0].embedding[:8])`
      : task === "rerank"
      ? `import requests

resp = requests.post(
    "${baseUrl}/rerank",
    headers={"Authorization": "Bearer YOUR_API_KEY"},
    json={
        "model": "${modelId}",
        "query": "What is the capital of France?",
        "documents": ["Paris is the capital of France.", "Berlin is in Germany."],
    },
)
print(resp.json())`
      : `from openai import OpenAI

client = OpenAI(
    base_url="${baseUrl}",
    api_key="YOUR_API_KEY",
)

response = client.chat.completions.create(
    model="${modelId}",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")`;

  const modeTabs = ([
    { id: "chat" as const, icon: MessageSquare, labelKey: "dash.serverless.try_chat", show: isPreset },
    { id: "job" as const, icon: Terminal, labelKey: "dash.serverless.try_job", show: true },
    { id: "snippets" as const, icon: Code2, labelKey: "dash.serverless.try_snippets", show: true },
  ]).filter((m) => m.show);

  return (
    <ServerlessPanel className="p-4 sm:p-5 space-y-4">
      <ServerlessSegmentedTabs tabs={modeTabs} value={mode} onChange={setMode} label={t} />

      {mode === "chat" && isPreset && (
        <div className="space-y-3">
          <Input
            value={prompt}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setPrompt(e.target.value)}
            placeholder={t("dash.serverless.prompt_ph")}
          />
          <div className="flex gap-2">
            <Button onClick={() => runChat(true)} disabled={streaming || !canWrite}>
              {streaming ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
              {t("dash.serverless.stream")}
            </Button>
            <Button variant="outline" onClick={() => runChat(false)} disabled={streaming || !canWrite}>
              {t("dash.serverless.sync")}
            </Button>
            {streaming && (
              <Button variant="ghost" onClick={stopStream}>
                <Square className="h-4 w-4" /> {t("dash.serverless.stop")}
              </Button>
            )}
          </div>
        </div>
      )}

      {mode === "job" && (
        <div className="space-y-3">
          <textarea
            className="w-full min-h-[100px] rounded-lg border border-border bg-background px-3 py-2 font-mono text-xs"
            value={jobPayload}
            onChange={(e) => setJobPayload(e.target.value)}
          />
          <div className="flex gap-2">
            <Button onClick={runJob} disabled={streaming || !canWrite}>
              {streaming ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
              {t("dash.serverless.run_job")}
            </Button>
            {streaming && (
              <Button variant="ghost" onClick={stopStream}>
                <Square className="h-4 w-4" /> {t("dash.serverless.stop")}
              </Button>
            )}
          </div>
        </div>
      )}

      {mode === "snippets" && (
        <div className="space-y-4">
          <SnippetBlock title="cURL" code={curlSnippet} />
          {isPreset && <SnippetBlock title="OpenAI Python SDK" code={openaiSnippet} />}
          <div className="text-xs text-text-muted">
            <span>{t("dash.serverless.openai_base")}: </span>
            <CopyableText text={baseUrl} />
          </div>
        </div>
      )}

      {(mode === "chat" || mode === "job") && (
        <div className="relative">
          <pre
            ref={outputRef}
            className="min-h-[160px] max-h-[320px] overflow-auto rounded-xl border border-border bg-[#0d1117] p-4 font-mono text-xs text-emerald-300/90"
          >
            {output || (streaming ? t("dash.serverless.waiting") : t("dash.serverless.output_empty"))}
          </pre>
          {streaming && (
            <Badge variant="info" className="absolute top-2 right-2 text-[10px] animate-pulse">
              LIVE
            </Badge>
          )}
        </div>
      )}
    </ServerlessPanel>
  );
}

function SnippetBlock({ title, code }: { title: string; code: string }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };
  return (
    <div className="rounded-xl border border-border overflow-hidden">
      <div className="flex items-center justify-between px-3 py-2 bg-surface-hover border-b border-border">
        <span className="text-xs font-medium">{title}</span>
        <button type="button" onClick={copy} className="text-text-muted hover:text-text-primary">
          {copied ? <Check className="h-3.5 w-3.5 text-emerald" /> : <Copy className="h-3.5 w-3.5" />}
        </button>
      </div>
      <pre className="p-3 text-xs font-mono overflow-x-auto bg-[#0d1117] text-text-secondary">{code}</pre>
    </div>
  );
}