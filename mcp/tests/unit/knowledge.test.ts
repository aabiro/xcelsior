import { describe, expect, it } from "vitest";
import {
  parseDocsIndex,
  scoreRecord,
  searchRecords,
  type KnowledgeRecord,
} from "../../src/tools/knowledge.js";
import { TOOL_CONTRACTS } from "../../src/tools/contracts.js";
import { TOOL_DESCRIPTIONS } from "../../src/tools/descriptions.js";
import { loadConfig } from "../../src/config.js";

const DOCS = "https://docs.xcelsior.ca";

const SAMPLE_INDEX = `# Xcelsior Docs

## Docs

- [Introduction](https://xcelsior.docs.buildwithfern.com/introduction.md): Distributed GPU compute.
- [Billing](https://xcelsior.docs.buildwithfern.com/billing.md): CAD-native billing with tax support.
- [Trust & Controls](https://xcelsior.docs.buildwithfern.com/compliance.md): What the platform enforces.
- [Introduction](https://xcelsior.docs.buildwithfern.com/introduction.md): duplicate line
not a list item
- [Broken](not-a-url): should be skipped
`;

function record(overrides: Partial<KnowledgeRecord> = {}): KnowledgeRecord {
  return {
    id: "docs:billing",
    title: "Billing",
    url: `${DOCS}/billing`,
    summary: "CAD-native billing with 13-province tax support",
    load: async () => "body",
    ...overrides,
  };
}

describe("company-knowledge corpus", () => {
  it("is off unless deliberately enabled", () => {
    // X1.11: optional, and must never delay the base plugin submission — so
    // the reviewed surface does not change until someone opts in.
    const saved = process.env.XCELSIOR_MCP_COMPANY_KNOWLEDGE;
    delete process.env.XCELSIOR_MCP_COMPANY_KNOWLEDGE;
    try {
      expect(loadConfig().companyKnowledge).toBe(false);
      process.env.XCELSIOR_MCP_COMPANY_KNOWLEDGE = "1";
      expect(loadConfig().companyKnowledge).toBe(true);
    } finally {
      if (saved === undefined) delete process.env.XCELSIOR_MCP_COMPANY_KNOWLEDGE;
      else process.env.XCELSIOR_MCP_COMPANY_KNOWLEDGE = saved;
    }
  });

  it("derives the page list from what the docs site publishes", () => {
    const records = parseDocsIndex(SAMPLE_INDEX, DOCS);
    expect(records.map((r) => r.id)).toEqual([
      "docs:introduction", "docs:billing", "docs:compliance",
    ]);
    expect(records[0].title).toBe("Introduction");
    expect(records[1].summary).toContain("CAD-native");
  });

  it("cites the human-readable page on our own host, not the .md on the CDN", () => {
    // The url is what a reader clicks. A `.md` on the hosting provider's
    // domain is neither ours nor pleasant to read.
    for (const entry of parseDocsIndex(SAMPLE_INDEX, DOCS)) {
      expect(entry.url.startsWith(`${DOCS}/`), entry.id).toBe(true);
      expect(entry.url.endsWith(".md"), entry.id).toBe(false);
      expect(entry.url).not.toContain("buildwithfern");
    }
  });

  it("returns absolute https URLs with no fragment", () => {
    for (const entry of parseDocsIndex(SAMPLE_INDEX, DOCS)) {
      const url = new URL(entry.url);
      expect(url.protocol).toBe("https:");
      expect(url.hash).toBe("");
      expect(url.hostname).toBeTruthy();
    }
  });

  it("requires every query term to match, so a miss returns nothing", () => {
    // A ranker that always returns something produces a citation to an
    // irrelevant page — a wrong answer with a link on it.
    const records = parseDocsIndex(SAMPLE_INDEX, DOCS);
    expect(searchRecords(records, "billing tax", 10).map((r) => r.id)).toEqual(["docs:billing"]);
    expect(searchRecords(records, "kubernetes helm charts", 10)).toEqual([]);
    expect(searchRecords(records, "", 10)).toEqual([]);
  });

  it("ranks a title match above a summary match", () => {
    const titled = record({ id: "docs:a", title: "Compliance", summary: "unrelated" });
    const summarised = record({ id: "docs:b", title: "unrelated", summary: "compliance rules" });
    expect(scoreRecord(titled, "compliance")).toBeGreaterThan(scoreRecord(summarised, "compliance"));
  });

  it("honours the result limit", () => {
    const many = Array.from({ length: 25 }, (_, i) =>
      record({ id: `docs:page-${i}`, title: `Billing topic ${i}` }));
    expect(searchRecords(many, "billing", 10)).toHaveLength(10);
  });
});

describe("company-knowledge contracts", () => {
  it("registers search and fetch as read-only tenant tools", () => {
    for (const name of ["search", "fetch"]) {
      expect(TOOL_CONTRACTS[name], name).toBeTruthy();
      expect(TOOL_CONTRACTS[name].annotations.readOnlyHint, name).toBe(true);
      expect(TOOL_CONTRACTS[name].annotations.destructiveHint, name).toBe(false);
      expect(TOOL_CONTRACTS[name].tenantClass, name).toBe("tenant");
      expect(TOOL_DESCRIPTIONS[name], name).toBeTruthy();
    }
  });

  it("tells the model that fetch ids come from search", () => {
    expect(TOOL_DESCRIPTIONS.fetch).toMatch(/after search/i);
    expect(TOOL_DESCRIPTIONS.search).toMatch(/\bid\b/);
  });
});

// Network-gated: GX1 requires that every URL company knowledge cites actually
// resolves. Run with XCELSIOR_MCP_KNOWLEDGE_URL_CHECK=1 (the scheduled
// conformance job sets it); skipped by default so unit tests stay offline.
const urlCheck = process.env.XCELSIOR_MCP_KNOWLEDGE_URL_CHECK === "1";
describe.skipIf(!urlCheck)("company-knowledge citations resolve", () => {
  it("every cited URL returns a success status", { timeout: 120_000 }, async () => {
    const index = await fetch(`${DOCS}/llms.txt`);
    expect(index.ok).toBe(true);
    const urls = [
      ...parseDocsIndex(await index.text(), DOCS).map((r) => r.url),
      "https://xcelsior.ca/llms.txt",
      "https://xcelsior.ca/pricing",
      "https://xcelsior.ca/gpus",
    ];
    const broken: string[] = [];
    for (const url of urls) {
      const response = await fetch(url, { redirect: "follow" });
      if (!response.ok) broken.push(`${url} → ${response.status}`);
    }
    expect(broken, "a citation a reader cannot open is worse than no citation").toEqual([]);
  });
});
