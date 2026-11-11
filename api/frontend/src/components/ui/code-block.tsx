"use client";

import { useMemo, useState } from "react";
import { Check, Copy } from "lucide-react";
import { cn } from "@/lib/utils";

export type CodeLanguage = "json" | "shell" | "python" | "text";

/**
 * Colour families, kept few so each one means something:
 *   key      names you set: JSON keys, keyword arguments, command flags
 *   string   quoted text and URLs
 *   value    literals that are not text: numbers, true/false/null, $VARS
 *   keyword  the language's own words, and the command being run
 *   comment  notes for the reader
 *   punct    structure: brackets, commas, pipes
 * The colours themselves are CSS variables (globals.css, marketing-theme.css).
 */
type TokenKind = "key" | "string" | "value" | "keyword" | "comment" | "punct";
type Token = { text: string; kind?: TokenKind };
type Rule = [RegExp, TokenKind | undefined];

const JSON_RULES: Rule[] = [
  [/"(?:\\.|[^"\\])*"(?=\s*:)/y, "key"],
  [/"(?:\\.|[^"\\])*"/y, "string"],
  [/(?:true|false|null)(?!\w)/y, "value"],
  [/-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/y, "value"],
  [/[{}[\],:]/y, "punct"],
];

const PYTHON_RULES: Rule[] = [
  [/#.*/y, "comment"],
  [/"""[\s\S]*?"""|'(?:\\.|[^'\\\n])*'|"(?:\\.|[^"\\\n])*"/y, "string"],
  [/(?:from|import|as|def|return|for|in|if|elif|else|while|with|class|try|except|finally|raise|not|and|or|is|lambda|async|await|pass|yield)(?!\w)/y, "keyword"],
  [/(?:True|False|None)(?!\w)/y, "value"],
  [/\d+(?:\.\d+)?/y, "value"],
  [/[A-Za-z_]\w*(?==[^=])/y, "key"],
  [/[A-Za-z_]\w*/y, undefined],
  [/[()[\]{},.:=]/y, "punct"],
];

// A shell word is everything up to whitespace or an operator, so `-H` is a
// flag only where a word starts and `xcelsior-gpu/skill` stays one word.
const SHELL_RULES: Rule[] = [
  [/'[^']*'|"(?:\\.|[^"\\])*"/y, "string"],
  [/\$\{?\w+\}?/y, "value"],
  [/\\$/my, "punct"],
  [/&&|\|\||[|;&]/y, "punct"],
  [/--?[A-Za-z][\w-]*/y, "key"],
  [/[^\s'"|;&\\$]+/y, undefined],
];

function scan(code: string, rules: Rule[]): Token[] {
  const tokens: Token[] = [];
  let plain = "";
  let pos = 0;
  outer: while (pos < code.length) {
    for (const [pattern, kind] of rules) {
      pattern.lastIndex = pos;
      const match = pattern.exec(code);
      if (match && match[0]) {
        if (plain) tokens.push({ text: plain });
        plain = "";
        tokens.push({ text: match[0], kind });
        pos += match[0].length;
        continue outer;
      }
    }
    plain += code[pos];
    pos += 1;
  }
  if (plain) tokens.push({ text: plain });
  return tokens;
}

function scanShell(code: string): Token[] {
  const tokens: Token[] = [];
  let continued = false;
  for (const [index, line] of code.split("\n").entries()) {
    if (index > 0) tokens.push({ text: "\n" });
    if (!continued && line.trimStart().startsWith("#")) {
      tokens.push({ text: line, kind: "comment" });
      continue;
    }
    // The first bare word of a command, and of each one after a pipe, is the program.
    let expectCommand = !continued;
    for (const token of scan(line, SHELL_RULES)) {
      if (token.kind === "punct" && token.text !== "\\") expectCommand = true;
      else if (token.kind === undefined && token.text.trim()) {
        if (expectCommand) token.kind = "keyword";
        else if (token.text.includes("://")) token.kind = "string";
        expectCommand = false;
      } else if (token.kind) expectCommand = false;
      tokens.push(token);
    }
    continued = line.trimEnd().endsWith("\\");
  }
  return tokens;
}

export function tokenizeCode(code: string, language: CodeLanguage): Token[] {
  if (language === "json") return scan(code, JSON_RULES);
  if (language === "python") return scan(code, PYTHON_RULES);
  if (language === "shell") return scanShell(code);
  return [{ text: code }];
}

function languageFor(filename: string | undefined, code: string): CodeLanguage {
  const name = (filename ?? "").toLowerCase();
  if (/\.json\b/.test(name)) return "json";
  if (/\.py\b/.test(name)) return "python";
  if (/\.sh\b/.test(name) || name === "terminal") return "shell";
  return /^\s*[{[]/.test(code) ? "json" : "text";
}

/** Code card: a filename bar with inline copy, and the code coloured by token family. */
export function CodeBlock({
  filename,
  code,
  language,
  onCopy,
  className,
}: {
  filename?: string;
  code: string;
  /** Inferred from the filename (or a leading `{`) when omitted. */
  language?: CodeLanguage;
  /** Optional side-effect after copy (e.g. a toast). Clipboard write is handled here. */
  onCopy?: () => void;
  className?: string;
}) {
  const [copied, setCopied] = useState(false);
  const tokens = useMemo(
    () => tokenizeCode(code, language ?? languageFor(filename, code)),
    [code, filename, language],
  );
  const copy = () => {
    void navigator.clipboard.writeText(code);
    setCopied(true);
    onCopy?.();
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <div
      className={cn(
        "overflow-hidden rounded-xl border border-border bg-surface shadow-sm",
        className,
      )}
    >
      <div className="flex items-center justify-between gap-2 border-b border-border bg-surface-hover px-3 py-2">
        <span className="min-w-0 truncate pl-1 font-mono text-[11px] text-text-muted">{filename}</span>
        <button
          type="button"
          onClick={copy}
          className="inline-flex shrink-0 items-center gap-1 rounded-md px-2 py-1 text-[11px] text-text-muted transition-colors hover:bg-surface hover:text-text-primary"
        >
          {copied ? <Check className="h-3 w-3 text-emerald" /> : <Copy className="h-3 w-3" />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className="code-syntax overflow-x-auto p-4 font-mono text-xs leading-relaxed text-text-primary">
        {tokens.map((token, i) =>
          token.kind ? (
            <span key={i} className={`tok-${token.kind}`}>
              {token.text}
            </span>
          ) : (
            token.text
          ),
        )}
      </pre>
    </div>
  );
}
