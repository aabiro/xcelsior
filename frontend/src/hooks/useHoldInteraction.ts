"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export interface UseHoldInteractionOptions {
  durationMs?: number;
  /** Cancel hold if pointer moves more than this many pixels. */
  moveTolerancePx?: number;
  enabled?: boolean;
  onArmed?: () => void;
  onHoldStart?: () => void;
  onHoldCancel?: () => void;
}

export function useHoldInteraction({
  durationMs = 720,
  moveTolerancePx = 14,
  enabled = true,
  onArmed,
  onHoldStart,
  onHoldCancel,
}: UseHoldInteractionOptions = {}) {
  const [progress, setProgress] = useState(0);
  const [isHolding, setIsHolding] = useState(false);
  const [isArmed, setIsArmed] = useState(false);
  const startRef = useRef<number | null>(null);
  const originRef = useRef({ x: 0, y: 0 });
  const activePointerRef = useRef<number | null>(null);
  const rafRef = useRef<number | null>(null);
  const isArmedRef = useRef(false);
  const onArmedRef = useRef(onArmed);
  const onHoldStartRef = useRef(onHoldStart);
  const onHoldCancelRef = useRef(onHoldCancel);

  useEffect(() => {
    onArmedRef.current = onArmed;
    onHoldStartRef.current = onHoldStart;
    onHoldCancelRef.current = onHoldCancel;
  }, [onArmed, onHoldStart, onHoldCancel]);

  useEffect(() => {
    isArmedRef.current = isArmed;
  }, [isArmed]);

  const stopRaf = useCallback(() => {
    if (rafRef.current != null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, []);

  const cancelHold = useCallback(() => {
    if (!isHolding && startRef.current == null) return;
    stopRaf();
    startRef.current = null;
    activePointerRef.current = null;
    setIsHolding(false);
    if (!isArmedRef.current) setProgress(0);
    onHoldCancelRef.current?.();
  }, [isHolding, stopRaf]);

  const disarm = useCallback(() => {
    stopRaf();
    startRef.current = null;
    activePointerRef.current = null;
    setIsHolding(false);
    setIsArmed(false);
    isArmedRef.current = false;
    setProgress(0);
  }, [stopRaf]);

  const onPointerDown = useCallback((e: React.PointerEvent<HTMLElement>) => {
    if (!enabled || isArmedRef.current) return;
    if (e.button !== 0) return;
    if (activePointerRef.current != null) return;

    activePointerRef.current = e.pointerId;
    originRef.current = { x: e.clientX, y: e.clientY };
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {
      /* noop, capture optional */
    }

    setIsHolding(true);
    startRef.current = performance.now();
    onHoldStartRef.current?.();

    const tick = (now: number) => {
      if (startRef.current == null || activePointerRef.current !== e.pointerId) return;
      const p = Math.min(1, (now - startRef.current) / durationMs);
      setProgress(p);
      if (p >= 1) {
        stopRaf();
        startRef.current = null;
        activePointerRef.current = null;
        setIsHolding(false);
        setIsArmed(true);
        isArmedRef.current = true;
        setProgress(1);
        try {
          navigator.vibrate?.(14);
        } catch {
          /* noop */
        }
        onArmedRef.current?.();
        return;
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  }, [durationMs, enabled, stopRaf]);

  const onPointerMove = useCallback((e: React.PointerEvent<HTMLElement>) => {
    if (activePointerRef.current !== e.pointerId || startRef.current == null) return;
    const dx = e.clientX - originRef.current.x;
    const dy = e.clientY - originRef.current.y;
    if (dx * dx + dy * dy > moveTolerancePx * moveTolerancePx) {
      cancelHold();
    }
  }, [cancelHold, moveTolerancePx]);

  const releasePointer = useCallback((e: React.PointerEvent<HTMLElement>) => {
    if (activePointerRef.current !== e.pointerId) return;
    try {
      if (e.currentTarget.hasPointerCapture(e.pointerId)) {
        e.currentTarget.releasePointerCapture(e.pointerId);
      }
    } catch {
      /* noop */
    }
    activePointerRef.current = null;
    if (isArmedRef.current) return;
    stopRaf();
    startRef.current = null;
    setIsHolding(false);
    setProgress(0);
  }, [stopRaf]);

  useEffect(() => () => stopRaf(), [stopRaf]);

  return {
    progress,
    isHolding,
    isArmed,
    disarm,
    cancelHold,
    bind: {
      onPointerDown,
      onPointerMove,
      onPointerUp: releasePointer,
      onPointerCancel: releasePointer,
      onLostPointerCapture: releasePointer,
      style: { touchAction: "none" as const, userSelect: "none" as const, WebkitUserSelect: "none" as const },
    },
  };
}