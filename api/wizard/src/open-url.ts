// open-url.ts — Best-effort cross-platform "open this URL in the browser".
//
// Shared by the wizard flow and the post-flow deepener menu. Never throws;
// returns whether a launcher command succeeded. Skipped entirely when there's
// no usable display (CI / headless) so tests and remote boxes don't hang.

export async function openUrl(url: string): Promise<boolean> {
    // Don't attempt to spawn a browser in obviously headless contexts.
    if (process.env.XCELSIOR_NO_BROWSER === "1") return false;

    const { execFile } = await import("node:child_process");
    const { promisify } = await import("node:util");
    const exec = promisify(execFile);

    const cmds: [string, string[]][] = process.platform === "darwin"
        ? [["open", [url]]]
        : process.platform === "win32"
            ? [["cmd", ["/c", "start", "", url]]]
            : [["xdg-open", [url]], ["sensible-browser", [url]], ["x-www-browser", [url]]];

    for (const [cmd, args] of cmds) {
        try {
            await exec(cmd, args, { timeout: 5_000 });
            return true;
        } catch {
            continue;
        }
    }
    return false;
}
