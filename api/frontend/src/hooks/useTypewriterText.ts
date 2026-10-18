"use client";

import { useEffect, useRef, useState } from "react";

interface UseTypewriterTextOptions {
  animate: boolean;
  resetKey: string;
}

export function useTypewriterText(
  text: string,
  { animate, resetKey }: UseTypewriterTextOptions,
) {
  const [displayedText, setDisplayedText] = useState(() => (animate ? "" : text));
  const previousKeyRef = useRef(resetKey);

  useEffect(() => {
    if (previousKeyRef.current !== resetKey) {
      previousKeyRef.current = resetKey;
      const frameId = requestAnimationFrame(() => {
        setDisplayedText(animate ? "" : text);
      });
      return () => cancelAnimationFrame(frameId);
    }

    if (!text) {
      const frameId = requestAnimationFrame(() => {
        setDisplayedText("");
      });
      return () => cancelAnimationFrame(frameId);
    }

    if (!text.startsWith(displayedText)) {
      const frameId = requestAnimationFrame(() => {
        setDisplayedText(text);
      });
      return () => cancelAnimationFrame(frameId);
    }

    if (displayedText.length >= text.length) return;

    const remaining = text.length - displayedText.length;
    const step = animate
      ? remaining > 120
        ? 6
        : remaining > 60
          ? 4
          : remaining > 24
            ? 2
            : 1
      : remaining > 80
        ? 8
        : remaining > 30
          ? 4
          : 2;
    const delayMs = animate ? 26 : 12;

    const timerId = window.setTimeout(() => {
      setDisplayedText(text.slice(0, Math.min(text.length, displayedText.length + step)));
    }, delayMs);

    return () => window.clearTimeout(timerId);
  }, [animate, displayedText, resetKey, text]);

  return {
    displayedText,
    isTyping: displayedText.length < text.length,
  };
}
