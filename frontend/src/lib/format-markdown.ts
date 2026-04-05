import DOMPurify from "dompurify";

const ALLOWED_TAGS = ["pre", "code", "strong", "a", "br"];
const ALLOWED_ATTR = ["class", "href", "target", "rel"];

/**
 * Convert a subset of Markdown to sanitised HTML.
 *
 * Supports fenced code blocks, inline code, bold, links, and newlines.
 * Output is sanitised with DOMPurify to prevent XSS.
 */
export function formatMarkdown(text: string): string {
    const raw = text
        .replace(
            /```(\w*)\n?([\s\S]*?)```/g,
            '<pre class="bg-navy/60 rounded-lg p-3 my-2.5 text-xs overflow-x-auto border border-border/30 backdrop-blur-sm"><code>$2</code></pre>',
        )
        .replace(
            /`([^`]+)`/g,
            '<code class="bg-navy/40 rounded px-1.5 py-0.5 text-xs font-mono border border-border/20">$1</code>',
        )
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(
            /\[([^\]]+)\]\(([^)]+)\)/g,
            '<a href="$2" target="_blank" rel="noopener noreferrer" class="text-accent-cyan underline decoration-accent-cyan/30 hover:decoration-accent-cyan/80 transition-colors">$1</a>',
        )
        .replace(/\n/g, "<br />");

    return DOMPurify.sanitize(raw, {
        ALLOWED_TAGS,
        ALLOWED_ATTR,
    });
}
