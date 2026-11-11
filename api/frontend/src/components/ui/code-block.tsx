"use client";

import { useEffect, useMemo, useRef, useState } from "react";
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
 * The colours themselves are CSS variables (globals.css `.code-card`).
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

/**
 * The one code card: every snippet, config file and terminal command in the
 * app and on the site is drawn by this. A label bar (filename, or "terminal")
 * with Copy on the right, then the code, coloured by token family.
 *
 * Its look lives in globals.css (`.code-card`) as a handful of `--code-*`
 * variables; a surface that needs a different look (the marketing site) sets
 * the variables and leaves the rules alone.
 */
export function CodeBlock({
  filename,
  code,
  language,
  onCopy,
  className,
}: {
  /** The label in the bar: a filename, or "terminal" for commands. */
  filename?: string;
  code: string;
  /** Inferred from the filename (or a leading `{`) when omitted. */
  language?: CodeLanguage;
  /** Called after the code is on the clipboard (e.g. a toast). */
  onCopy?: () => void;
  className?: string;
}) {
  const [copied, setCopied] = useState(false);
  const resetTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => () => {
    if (resetTimer.current) clearTimeout(resetTimer.current);
  }, []);
  const tokens = useMemo(
    () => tokenizeCode(code, language ?? languageFor(filename, code)),
    [code, filename, language],
  );
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code);
    } catch {
      return; // No clipboard (insecure context, denied): say nothing rather than "Copied".
    }
    setCopied(true);
    onCopy?.();
    if (resetTimer.current) clearTimeout(resetTimer.current);
    resetTimer.current = setTimeout(() => setCopied(false), 2000);
  };
  return (
    <div className={cn("code-card", className)}>
      <div className="code-card-bar">
        <span className="code-card-label">{filename}</span>
        <button
          type="button"
          onClick={() => void copy()}
          className="code-card-copy"
          data-copied={copied || undefined}
          aria-label={copied ? "Copied" : filename ? `Copy ${filename}` : "Copy code"}
        >
          {copied ? <Check className="h-3 w-3" aria-hidden /> : <Copy className="h-3 w-3" aria-hidden />}
          <span aria-hidden>{copied ? "Copied" : "Copy"}</span>
        </button>
      </div>
      <pre className="code-card-body">
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
