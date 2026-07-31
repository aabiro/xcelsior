import { describe, expect, it } from "vitest";
import { formatMarkdown } from "@/lib/format-markdown";

describe("formatMarkdown", () => {
    it("wraps fenced code blocks in <pre><code>", () => {
        const html = formatMarkdown("```js\nconsole.log(1)\n```");
        expect(html).toContain("<pre");
        expect(html).toContain("<code>");
        expect(html).toContain("console.log(1)");
    });

    it("wraps inline code in <code>", () => {
        const html = formatMarkdown("use `npm install` here");
        expect(html).toContain("<code");
        expect(html).toContain("npm install");
        expect(html).toContain("</code>");
    });

    it("wraps bold text in <strong>", () => {
        const html = formatMarkdown("this is **bold** text");
        expect(html).toContain("<strong>bold</strong>");
    });

    it("converts markdown links to <a> tags", () => {
        const html = formatMarkdown("[click](https://example.com)");
        expect(html).toContain('<a href="https://example.com"');
        expect(html).not.toContain('target="_blank"');
        expect(html).toContain('rel="noopener noreferrer"');
        expect(html).toContain("click</a>");
    });

    it("converts newlines to <br>", () => {
        const html = formatMarkdown("line1\nline2");
        expect(html).toContain("<br");
    });

    it("neutralises script tags by escaping them", () => {
        const html = formatMarkdown('<script>alert("xss")</script>');
        // Escaped to inert text rather than deleted, so a code example
        // mentioning a script tag still reads correctly. The property that
        // matters is that no live tag reaches the DOM.
        expect(html).not.toContain("<script");
        expect(html).toContain("&lt;script&gt;");
    });

    it("strips event handler attributes", () => {
        const html = formatMarkdown('[link](javascript:alert(1))');
        expect(html).not.toContain("javascript:");
    });

    it("neutralises event handler attributes", () => {
        const html = formatMarkdown("plain text with <img onerror=alert(1)>");
        expect(html).not.toContain("<img");
        expect(html).toContain("&lt;img");
    });

    it("lets no executable markup reach the DOM", () => {
        const payloads = [
            "<script>alert(1)</script>",
            "<img src=x onerror=alert(1)>",
            "<svg/onload=alert(1)>",
            '<iframe src="javascript:alert(1)">',
            "[x](javascript:alert(1))",
            "<details open ontoggle=alert(1)>",
        ];
        for (const payload of payloads) {
            const el = document.createElement("div");
            el.innerHTML = formatMarkdown(payload);
            expect(el.querySelector("script"), payload).toBeNull();
            expect(el.querySelector("img,svg,iframe,details"), payload).toBeNull();
            for (const node of Array.from(el.querySelectorAll("*"))) {
                for (const attr of Array.from(node.attributes)) {
                    expect(attr.name.startsWith("on"), payload).toBe(false);
                    if (attr.name === "href") {
                        expect(attr.value.toLowerCase().startsWith("javascript:"), payload).toBe(
                            false,
                        );
                    }
                }
            }
        }
    });

    it("handles empty string", () => {
        expect(formatMarkdown("")).toBe("");
    });

    it("returns plain text unchanged when no markdown", () => {
        const html = formatMarkdown("hello world");
        expect(html).toBe("hello world");
    });
    // ── Block-level constructs the model actually emits ────────────────

    it("renders headings as capped heading tags", () => {
        expect(formatMarkdown("# Title")).toContain("<h3");
        expect(formatMarkdown("## Section")).toContain("<h4");
        expect(formatMarkdown("### Sub")).toContain("<h5");
        expect(formatMarkdown("## Section")).toContain("Section");
    });

    it("never emits h1 or h2, so chat cannot outrank page headings", () => {
        const html = formatMarkdown("# Big\n## Also big");
        expect(html).not.toContain("<h1");
        expect(html).not.toContain("<h2");
    });

    it("renders bullet lists", () => {
        const html = formatMarkdown("- one\n- two");
        expect(html).toContain("<ul");
        expect(html).toContain("<li>one</li>");
        expect(html).toContain("<li>two</li>");
    });

    it("renders numbered lists as <ol>", () => {
        const html = formatMarkdown("1. first\n2. second");
        expect(html).toContain("<ol");
        expect(html).toContain("<li>first</li>");
        expect(html).toContain("<li>second</li>");
    });

    it("does not leak list markers as literal text", () => {
        const html = formatMarkdown("- alpha\n- beta");
        expect(html).not.toContain("- alpha");
    });

    it("renders tables with headers and rows", () => {
        const html = formatMarkdown("| GPU | Rate |\n|-----|------|\n| 4090 | 1.20 |");
        expect(html).toContain("<table");
        expect(html).toContain("<th");
        expect(html).toContain("GPU");
        expect(html).toContain("<td");
        expect(html).toContain("4090");
    });

    it("renders blockquotes", () => {
        const html = formatMarkdown("> heads up");
        expect(html).toContain("<blockquote");
        expect(html).toContain("heads up");
    });

    it("renders horizontal rules", () => {
        expect(formatMarkdown("---")).toContain("<hr");
    });

    it("renders italics without breaking bold or identifiers", () => {
        expect(formatMarkdown("this is *italic*")).toContain("<em>italic</em>");
        expect(formatMarkdown("this is **bold**")).toContain("<strong>bold</strong>");
        expect(formatMarkdown("some_var_name")).toContain("some_var_name");
        expect(formatMarkdown("some_var_name")).not.toContain("<em>");
    });

    it("renders strikethrough", () => {
        expect(formatMarkdown("~~gone~~")).toContain("<del>gone</del>");
    });

    it("keeps markup inside code fences visible as text", () => {
        const html = formatMarkdown("```\n<script>alert(1)</script>\n```");
        expect(html).toContain("&lt;script&gt;");
        expect(html).not.toContain("<script>");
    });

    it("renders an unterminated fence while streaming", () => {
        const html = formatMarkdown("```py\nx = 1");
        expect(html).toContain("<pre");
        expect(html).toContain("x = 1");
    });

    it("handles a mixed document end to end", () => {
        const html = formatMarkdown(
            "## Costs\n\nHere are the **rates**:\n\n- 4090 at `1.20`\n- A100 at `3.50`\n\n> Prices are CAD.",
        );
        expect(html).toContain("<h4");
        expect(html).toContain("<strong>rates</strong>");
        expect(html).toContain("<ul");
        expect(html).toContain("<code");
        expect(html).toContain("<blockquote");
    });

    it("escapes raw html in prose instead of rendering it", () => {
        const html = formatMarkdown("a < b and c > d");
        expect(html).toContain("&lt;");
        expect(html).toContain("&gt;");
    });
});
