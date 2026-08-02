/**
 * ChatGPT company-knowledge `search` / `fetch` (adoption plan X1.11).
 *
 * Optional and off by default — the plan is explicit that this track must not
 * delay the base plugin submission, so enabling it is a deliberate act
 * (`XCELSIOR_MCP_COMPANY_KNOWLEDGE=1`) rather than something that quietly
 * widens the surface a directory already reviewed.
 *
 * Two contracts matter and neither is ours to redesign:
 *
 *  - `search(query)` returns `{results: [{id, title, url}]}`.
 *  - `fetch(id)` returns `{id, title, text, url, metadata}`.
 *
 * `url` must be **absolute and openable by a human** — it becomes the citation
 * a reader clicks, so a fragment, a relative path, or an API endpoint that
 * renders as JSON all fail the purpose. `id` carries our internal identifier
 * and is opaque to the caller.
 *
 * The corpus is assembled from four sources we already publish: the docs site's
 * own machine-readable index, `llms.txt`, the live pricing table, and the live
 * marketplace. Nothing here reads tenant data — this is public knowledge, and
 * treating it otherwise would make every citation unverifiable.
 */
import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { AuthUser } from "../auth/bearer.js";
import type { XcelsiorApiClient } from "../client/api.js";
import { TOOL_SCOPES, userHasScope, scopeUnion, describeScopeRequirement } from "../auth/scopes.js";
import { structuredResult } from "../lib/format.js";

export interface KnowledgeSources {
  /** Public marketing site. */
  siteUrl: string;
  /** Published documentation site. */
  docsUrl: string;
}

export interface KnowledgeRecord {
  id: string;
  title: string;
  /** Absolute, human-openable. */
  url: string;
  /** What search matches against and what a result preview shows. */
  summary: string;
  /** Full text, resolved lazily — search must not fetch every document. */
  load: () => Promise<string>;
}

const INDEX_TTL_MS = 15 * 60_000;
const FETCH_TIMEOUT_MS = 8_000;
const MAX_DOCUMENT_BYTES = 512 * 1024;

let indexCache: { at: number; records: KnowledgeRecord[] } | null = null;

export function resetKnowledgeCache(): void {
  indexCache = null;
}

async function getText(url: string): Promise<string> {
  const response = await fetch(url, { signal: AbortSignal.timeout(FETCH_TIMEOUT_MS) });
  if (!response.ok) throw new Error(`${url} returned HTTP ${response.status}`);
  const text = await response.text();
  return text.length > MAX_DOCUMENT_BYTES ? text.slice(0, MAX_DOCUMENT_BYTES) : text;
}

/**
 * Parse the docs site's own `llms.txt` index.
 *
 * Deriving the page list from what the docs site publishes — rather than a
 * hand-kept manifest here — means a new documentation page is searchable the
 * moment it ships, and a deleted one stops being cited. A hardcoded list would
 * be wrong within a release.
 */
export function parseDocsIndex(llmsTxt: string, docsUrl: string): KnowledgeRecord[] {
  const line = /^-\s*\[([^\]]+)\]\((https?:\/\/[^)\s]+?)\)\s*:?\s*(.*)$/;
  const records: KnowledgeRecord[] = [];
  const seen = new Set<string>();
  for (const raw of llmsTxt.split("\n")) {
    const match = line.exec(raw.trim());
    if (!match) continue;
    const [, title, href, summary] = match;
    let slug: string;
    try {
      slug = new URL(href).pathname.replace(/^\/+/, "").replace(/\.md$/, "");
    } catch {
      continue;
    }
    if (!slug || seen.has(slug)) continue;
    seen.add(slug);
    records.push({
      id: `docs:${slug}`,
      title: title.trim(),
      // The index links to `<page>.md`; the citation must point at the page a
      // person can read, and the site's own host, not the hosting provider's.
      url: `${docsUrl}/${slug}`,
      summary: summary.trim() || title.trim(),
      load: () => getText(`${docsUrl}/${slug}.md`),
    });
  }
  return records;
}

async function buildIndex(
  client: XcelsiorApiClient,
  sources: KnowledgeSources,
): Promise<KnowledgeRecord[]> {
  const records: KnowledgeRecord[] = [];

  try {
    records.push(...parseDocsIndex(await getText(`${sources.docsUrl}/llms.txt`), sources.docsUrl));
  } catch {
    // A docs outage degrades the corpus; it must not fail the whole tool.
  }

  records.push({
    id: "llms:txt",
    title: "Xcelsior llms.txt — machine-readable platform overview",
    url: `${sources.siteUrl}/llms.txt`,
    summary:
      "Condensed description of the Xcelsior GPU marketplace for agents: endpoints, " +
      "concepts, pricing model, and how to run work.",
    load: () => getText(`${sources.siteUrl}/llms.txt`),
  });

  records.push({
    id: "pricing:reference",
    title: "GPU pricing reference — CAD hourly rates by model",
    url: `${sources.siteUrl}/pricing`,
    summary:
      "Live on-demand and spot hourly rates in CAD for every GPU model Xcelsior offers, " +
      "with VRAM and tier.",
    load: async () => JSON.stringify(await client.get("/api/pricing/reference"), null, 2),
  });

  records.push({
    id: "marketplace:listings",
    title: "Marketplace listings — GPUs available now",
    url: `${sources.siteUrl}/gpus`,
    summary:
      "Current listings from independent hosts: GPU model, VRAM, region, " +
      "host reputation, and hourly rate.",
    load: async () => JSON.stringify(await client.post("/api/v2/marketplace/search", {}), null, 2),
  });

  return records;
}

async function knowledgeIndex(
  client: XcelsiorApiClient,
  sources: KnowledgeSources,
): Promise<KnowledgeRecord[]> {
  const now = Date.now();
  if (indexCache && now - indexCache.at < INDEX_TTL_MS) return indexCache.records;
  const records = await buildIndex(client, sources);
  indexCache = { at: now, records };
  return records;
}

function tokenize(value: string): string[] {
  return value.toLowerCase().match(/[a-z0-9]+/g) ?? [];
}

/**
 * Score a record against a query.
 *
 * Deliberately simple: title matches outrank summary matches, and every query
 * term must appear somewhere for a record to be a candidate. A fuzzy ranker
 * would return something for every query, which is worse than returning
 * nothing — a citation to an irrelevant page is a wrong answer with a link on
 * it.
 */
export function scoreRecord(record: KnowledgeRecord, query: string): number {
  const terms = [...new Set(tokenize(query))];
  if (!terms.length) return 0;
  const title = tokenize(record.title);
  const summary = tokenize(record.summary);
  const id = tokenize(record.id);
  let score = 0;
  let matched = 0;
  for (const term of terms) {
    const inTitle = title.some((word) => word.startsWith(term));
    const inSummary = summary.some((word) => word.startsWith(term));
    const inId = id.some((word) => word.startsWith(term));
    if (inTitle) score += 3;
    if (inId) score += 2;
    if (inSummary) score += 1;
    if (inTitle || inSummary || inId) matched += 1;
  }
  return matched === terms.length ? score : 0;
}

export function searchRecords(
  records: KnowledgeRecord[],
  query: string,
  limit: number,
): KnowledgeRecord[] {
  return records
    .map((record) => ({ record, score: scoreRecord(record, query) }))
    .filter((entry) => entry.score > 0)
    .sort((a, b) => b.score - a.score || a.record.id.localeCompare(b.record.id))
    .slice(0, limit)
    .map((entry) => entry.record);
}

const searchOutput = z.object({
  results: z.array(z.object({ id: z.string(), title: z.string(), url: z.string() })),
});
const fetchOutput = z.object({
  id: z.string(),
  title: z.string(),
  text: z.string(),
  url: z.string(),
  metadata: z.record(z.unknown()).optional(),
});

export function registerKnowledgeTools(
  server: McpServer,
  client: XcelsiorApiClient,
  sources: KnowledgeSources,
  user?: AuthUser,
): void {
  const denied = (tool: "search" | "fetch") => {
    const required = TOOL_SCOPES[tool];
    return userHasScope(user?.scopes, required)
      ? null
      : structuredResult(
          { ok: false, code: "insufficient_scope", required: scopeUnion(required) },
          `Access denied: requires ${describeScopeRequirement(required)}.`,
        );
  };

  server.registerTool(
    "search",
    {
      inputSchema: z.object({ query: z.string().min(1).max(512) }),
      outputSchema: searchOutput,
    },
    async ({ query }) => {
      const scopeError = denied("search");
      if (scopeError) return scopeError;
      try {
        const records = await knowledgeIndex(client, sources);
        return structuredResult({
          results: searchRecords(records, query, 10).map(({ id, title, url }) => ({
            id, title, url,
          })),
        });
      } catch (error) {
        return structuredResult({ ok: false, error: String(error), results: [] });
      }
    },
  );

  server.registerTool(
    "fetch",
    {
      inputSchema: z.object({ id: z.string().min(1).max(256) }),
      outputSchema: fetchOutput,
    },
    async ({ id }) => {
      const scopeError = denied("fetch");
      if (scopeError) return scopeError;
      try {
        const records = await knowledgeIndex(client, sources);
        const record = records.find((entry) => entry.id === id);
        if (!record) {
          return structuredResult({
            ok: false,
            code: "not_found",
            error: `No document with id ${id}. Call search first and use an id it returned.`,
          });
        }
        return structuredResult({
          id: record.id,
          title: record.title,
          text: await record.load(),
          url: record.url,
          metadata: { source: record.id.split(":")[0], summary: record.summary },
        });
      } catch (error) {
        return structuredResult({ ok: false, error: String(error) });
      }
    },
  );
}
