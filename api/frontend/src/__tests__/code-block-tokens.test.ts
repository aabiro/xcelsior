import { describe, expect, it } from "vitest";
import { tokenizeCode } from "@/components/ui/code-block";

const kinds = (code: string, language: Parameters<typeof tokenizeCode>[1]) =>
  tokenizeCode(code, language)
    .filter((t) => t.kind)
    .map((t) => `${t.kind}:${t.text}`);

describe("code card token families", () => {
  it("never changes the text it colours", () => {
    const samples: Array<[string, Parameters<typeof tokenizeCode>[1]]> = [
      ['{"a": {"url": "https://x.ca/mcp", "n": -1.5, "ok": true}}', "json"],
      ["curl -s -X POST 'https://x.ca/oauth/token' \\\n  -H 'A: b' | jq .", "shell"],
      ['client = OpenAI(\n    # note\n    base_url="https://x",\n)\nprint(resp.choices[0])', "python"],
    ];
    for (const [code, language] of samples) {
      expect(tokenizeCode(code, language).map((t) => t.text).join("")).toBe(code);
    }
  });

  it("separates JSON keys from string values and literals", () => {
    expect(kinds('{"type": "http", "n": 2, "on": true}', "json")).toEqual([
      "punct:{", "key:\"type\"", "punct::", "string:\"http\"", "punct:,",
      "key:\"n\"", "punct::", "value:2", "punct:,",
      "key:\"on\"", "punct::", "value:true", "punct:}",
    ]);
  });

  it("marks the program, its flags and quoted arguments in a shell command", () => {
    expect(kinds("npx skills add xcelsior-gpu/skill", "shell")).toEqual(["keyword:npx"]);
    expect(kinds("curl -s -X POST 'https://x.ca/t' \\\n  -H 'A: b' | jq .", "shell")).toEqual([
      "keyword:curl", "key:-s", "key:-X", "string:'https://x.ca/t'", "punct:\\",
      "key:-H", "string:'A: b'", "punct:|", "keyword:jq",
    ]);
  });

  it("marks Python keywords, keyword arguments, strings and comments", () => {
    const found = kinds('from openai import OpenAI\nc = OpenAI(\n    # two\n    api_key="K",\n)', "python");
    expect(found).toContain("keyword:from");
    expect(found).toContain("keyword:import");
    expect(found).toContain("comment:# two");
    expect(found).toContain("key:api_key");
    expect(found).toContain('string:"K"');
    expect(found).not.toContain("key:c");
  });
});
