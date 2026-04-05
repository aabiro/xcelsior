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
        expect(html).toContain('target="_blank"');
        expect(html).toContain('rel="noopener noreferrer"');
        expect(html).toContain("click</a>");
    });

    it("converts newlines to <br>", () => {
        const html = formatMarkdown("line1\nline2");
        expect(html).toContain("<br");
    });

    it("strips disallowed tags via DOMPurify", () => {
        const html = formatMarkdown('<script>alert("xss")</script>');
        expect(html).not.toContain("<script");
        expect(html).not.toContain("alert");
    });

    it("strips event handler attributes", () => {
        const html = formatMarkdown('[link](javascript:alert(1))');
        expect(html).not.toContain("javascript:");
    });

    it("strips disallowed attributes", () => {
        const html = formatMarkdown("plain text with <img onerror=alert(1)>");
        expect(html).not.toContain("onerror");
        expect(html).not.toContain("<img");
    });

    it("handles empty string", () => {
        expect(formatMarkdown("")).toBe("");
    });

    it("returns plain text unchanged when no markdown", () => {
        const html = formatMarkdown("hello world");
        expect(html).toBe("hello world");
    });
});
