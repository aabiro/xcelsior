// use-terminal-width.ts — shared hook tracking terminal width with resize.

import { useEffect, useState } from "react";
import { useStdout } from "ink";

/** Track terminal column count, updating on resize. */
export function useTerminalWidth(): number {
    const { stdout } = useStdout();
    const [width, setWidth] = useState(stdout?.columns ?? 80);
    useEffect(() => {
        if (!stdout) return;
        const onResize = () => setWidth(stdout.columns ?? 80);
        stdout.on("resize", onResize);
        return () => {
            stdout.off?.("resize", onResize);
        };
    }, [stdout]);
    return width;
}
