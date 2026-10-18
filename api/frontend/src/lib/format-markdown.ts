import DOMPurify from "dompurify";

const ALLOWED_TAGS = [
    "pre",
    "code",
    "strong",
    "em",
    "del",
    "a",
    "br",
    "h3",
    "h4",
    "h5",
    "ul",
    "ol",
    "li",
    "blockquote",
    "hr",
    "table",
    "thead",
    "tbody",
    "tr",
    "th",
    "td",
];
const ALLOWED_ATTR = ["class", "href", "target", "rel"];

const CODE_CLASS =
    "bg-navy/60 rounded-lg p-3 my-2.5 text-xs overflow-x-auto border border-border/30 backdrop-blur-sm";
const INLINE_CODE_CLASS =
    "bg-navy/40 rounded px-1.5 py-0.5 text-xs font-mono border border-border/20";
const LINK_CLASS =
    "text-accent-cyan underline decoration-accent-cyan/30 hover:decoration-accent-cyan/80 transition-colors";
const HEADING_CLASS: Record<string, string> = {
    h3: "font-semibold text-sm mt-3 mb-1.5 first:mt-0",
    h4: "font-semibold text-sm mt-2.5 mb-1 first:mt-0",
    h5: "font-semibold text-xs uppercase tracking-wide opacity-80 mt-2 mb-1 first:mt-0",
};
const LIST_CLASS = "my-1.5 ml-4 space-y-0.5 list-outside";
const TABLE_CLASS = "my-2.5 w-full text-xs border-collapse";
const CELL_CLASS = "border border-border/30 px-2 py-1 text-left align-top";

function escapeHtml(text: string): string {
    return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/** Apply span-level markdown to already-escaped text. */
function formatInline(escaped: string): string {
    return (
        escaped
            .replace(/`([^`]+)`/g, `<code class="${INLINE_CODE_CLASS}">$1</code>`)
            .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
            .replace(/~~(.+?)~~/g, "<del>$1</del>")
            // Single * / _ emphasis. Guarded so it neither re-consumes the **
            // handled above nor italicises intra-word underscores in
            // identifiers like some_var_name.
            .replace(/(^|[^*\w])\*(?!\s)([^*]+?)\*(?![*\w])/g, "$1<em>$2</em>")
            .replace(/(^|[^_\w])_(?!\s)([^_]+?)_(?![_\w])/g, "$1<em>$2</em>")
            .replace(
                /\[([^\]]+)\]\(([^)\s]+)\)/g,
                `<a href="$2" rel="noopener noreferrer" class="${LINK_CLASS}">$1</a>`,
            )
    );
}

const FENCE_RE = /^\s*```/;
const HEADING_RE = /^(#{1,6})\s+(.*)$/;
const ULIST_RE = /^\s*[-*+]\s+(.*)$/;
const OLIST_RE = /^\s*\d+[.)]\s+(.*)$/;
const QUOTE_RE = /^\s*>\s?(.*)$/;
const RULE_RE = /^\s*([-*_])(\s*\1){2,}\s*$/;
const TABLE_DIVIDER_RE = /^\s*\|?[\s:-]*-[\s|:-]*\|?\s*$/;

function splitRow(line: string): string[] {
    return line
        .replace(/^\s*\|/, "")
        .replace(/\|\s*$/, "")
        .split("|")
        .map((cell) => cell.trim());
}

/**
 * Convert Markdown to sanitised HTML for chat messages.
 *
 * Handles the block constructs models actually emit — headings, bullet and
 * numbered lists, tables, blockquotes, horizontal rules and fenced code —
 * plus inline bold, italic, strikethrough, code and links.
 *
 * Plain prose is deliberately not wrapped in <p>, so a one-line answer stays
 * a bare string and chat bubbles keep their own spacing. Input is HTML-escaped
 * before formatting, so markup inside a code fence renders as visible text
 * instead of being stripped; output is still sanitised with DOMPurify.
 */
export function formatMarkdown(text: string): string {
    const lines = text.replace(/\r\n?/g, "\n").split("\n");
    const blocks: string[] = [];
    let i = 0;

    while (i < lines.length) {
        const line = lines[i];

        // ── Fenced code ────────────────────────────────────────────────
        if (FENCE_RE.test(line)) {
            const body: string[] = [];
            i++;
            while (i < lines.length && !FENCE_RE.test(lines[i])) {
                body.push(lines[i]);
                i++;
            }
            i++; // closing fence; absent while a response is still streaming
            blocks.push(
                `<pre class="${CODE_CLASS}"><code>${escapeHtml(body.join("\n"))}</code></pre>`,
            );
            continue;
        }

        // ── Horizontal rule ────────────────────────────────────────────
        if (RULE_RE.test(line)) {
            blocks.push('<hr class="my-3 border-border/30" />');
            i++;
            continue;
        }

        // ── Heading ────────────────────────────────────────────────────
        const heading = line.match(HEADING_RE);
        if (heading) {
            // Chat bubbles are not documents: cap at h3 so a model-emitted h1
            // cannot outrank the page's own heading hierarchy.
            const tag = heading[1].length <= 1 ? "h3" : heading[1].length === 2 ? "h4" : "h5";
            blocks.push(
                `<${tag} class="${HEADING_CLASS[tag]}">${formatInline(
                    escapeHtml(heading[2].trim()),
                )}</${tag}>`,
            );
            i++;
            continue;
        }

        // ── Table ──────────────────────────────────────────────────────
        if (
            line.includes("|") &&
            i + 1 < lines.length &&
            lines[i + 1].includes("-") &&
            TABLE_DIVIDER_RE.test(lines[i + 1])
        ) {
            const header = splitRow(line);
            i += 2;
            const rows: string[][] = [];
            while (i < lines.length && lines[i].includes("|") && lines[i].trim()) {
                rows.push(splitRow(lines[i]));
                i++;
            }
            const head = header
                .map(
                    (cell) =>
                        `<th class="${CELL_CLASS} font-semibold">${formatInline(
                            escapeHtml(cell),
                        )}</th>`,
                )
                .join("");
            const body = rows
                .map(
                    (row) =>
                        `<tr>${row
                            .map(
                                (cell) =>
                                    `<td class="${CELL_CLASS}">${formatInline(
                                        escapeHtml(cell),
                                    )}</td>`,
                            )
                            .join("")}</tr>`,
                )
                .join("");
            blocks.push(
                `<table class="${TABLE_CLASS}"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`,
            );
            continue;
        }

        // ── Lists ──────────────────────────────────────────────────────
        if (ULIST_RE.test(line) || OLIST_RE.test(line)) {
            const ordered = OLIST_RE.test(line) && !ULIST_RE.test(line);
            const re = ordered ? OLIST_RE : ULIST_RE;
            const items: string[] = [];
            while (i < lines.length) {
                const match = lines[i].match(re);
                if (!match) break;
                items.push(`<li>${formatInline(escapeHtml(match[1]))}</li>`);
                i++;
            }
            const tag = ordered ? "ol" : "ul";
            const style = ordered ? "list-decimal" : "list-disc";
            blocks.push(`<${tag} class="${LIST_CLASS} ${style}">${items.join("")}</${tag}>`);
            continue;
        }

        // ── Blockquote ─────────────────────────────────────────────────
        if (QUOTE_RE.test(line)) {
            const quoted: string[] = [];
            while (i < lines.length) {
                const match = lines[i].match(QUOTE_RE);
                if (!match) break;
                quoted.push(formatInline(escapeHtml(match[1])));
                i++;
            }
            blocks.push(
                `<blockquote class="my-2 border-l-2 border-border/50 pl-3 opacity-90">${quoted.join(
                    "<br />",
                )}</blockquote>`,
            );
            continue;
        }

        // ── Paragraph ──────────────────────────────────────────────────
        const paragraph: string[] = [];
        while (i < lines.length) {
            const current = lines[i];
            if (
                FENCE_RE.test(current) ||
                HEADING_RE.test(current) ||
                ULIST_RE.test(current) ||
                OLIST_RE.test(current) ||
                QUOTE_RE.test(current) ||
                RULE_RE.test(current)
            ) {
                break;
            }
            paragraph.push(formatInline(escapeHtml(current)));
            i++;
        }
        if (paragraph.length) blocks.push(paragraph.join("<br />"));
    }

    return DOMPurify.sanitize(blocks.join(""), { ALLOWED_TAGS, ALLOWED_ATTR });
}
