/**
 * Pinch the web terminal to resize its text on a touch screen.
 *
 * nvidia-smi prints an 89-column table, and a phone held upright fits about
 * 36 columns at the default size, so the table wrapped into a mess. Pinching
 * two fingers on the terminal shrinks or grows the text, and with it the
 * column count; the shell is told the new size like any other resize. The
 * size is remembered in this browser. Only touch is handled: a mouse, a
 * trackpad and the desktop layout are unchanged.
 */

export const TERMINAL_FONT_DEFAULT = 13;
// 6px still reads on a phone's 3x screen and fits nvidia-smi upright.
export const TERMINAL_FONT_MIN = 6;
export const TERMINAL_FONT_MAX = 24;
const STORAGE_KEY = "xcelsior.terminal.fontSize";

export function clampFontSize(size: number): number {
  if (!Number.isFinite(size)) return TERMINAL_FONT_DEFAULT;
  return Math.min(TERMINAL_FONT_MAX, Math.max(TERMINAL_FONT_MIN, Math.round(size)));
}

/** The size for a pinch that started at `startSize` with the fingers `startDistance` apart. */
export function pinchFontSize(startSize: number, startDistance: number, distance: number): number {
  if (!(startDistance > 0) || !(distance > 0)) return clampFontSize(startSize);
  return clampFontSize(startSize * (distance / startDistance));
}

export function storedFontSize(): number {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    const size = raw === null ? Number.NaN : Number.parseInt(raw, 10);
    return Number.isFinite(size) ? clampFontSize(size) : TERMINAL_FONT_DEFAULT;
  } catch {
    return TERMINAL_FONT_DEFAULT;
  }
}

export function storeFontSize(size: number): void {
  try {
    if (size === TERMINAL_FONT_DEFAULT) window.localStorage.removeItem(STORAGE_KEY);
    else window.localStorage.setItem(STORAGE_KEY, String(clampFontSize(size)));
  } catch {
    // Storage refused (private browsing): the size lasts for this page.
  }
}

export interface PinchFont {
  get(): number;
  set(size: number): void;
  /** The pinch ended at `size`. */
  done(size: number): void;
}

function fingerGap(touches: TouchList): number {
  const a = touches[0];
  const b = touches[1];
  return Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
}

/**
 * Listens on `el` in the capture phase, so a two-finger touch reaches the
 * pinch before xterm turns it into a scroll, and Safari does not zoom the
 * page instead. One finger is left to xterm. Returns a function that stops
 * listening.
 */
export function attachPinchToResize(el: HTMLElement, font: PinchFont): () => void {
  let start: { size: number; gap: number } | null = null;

  const begin = (ev: TouchEvent) => {
    if (ev.touches.length !== 2) return;
    start = { size: font.get(), gap: fingerGap(ev.touches) };
    ev.stopPropagation();
  };
  const move = (ev: TouchEvent) => {
    if (!start || ev.touches.length !== 2) return;
    ev.preventDefault();
    ev.stopPropagation();
    const next = pinchFontSize(start.size, start.gap, fingerGap(ev.touches));
    if (next !== font.get()) font.set(next);
  };
  const end = (ev: TouchEvent) => {
    if (!start || ev.touches.length >= 2) return;
    start = null;
    font.done(font.get());
  };
  // Safari's own pinch gesture, which would zoom the whole page.
  const noPageZoom = (ev: Event) => ev.preventDefault();

  const options: AddEventListenerOptions = { capture: true, passive: false };
  const listeners: [string, EventListener][] = [
    ["touchstart", begin as EventListener],
    ["touchmove", move as EventListener],
    ["touchend", end as EventListener],
    ["touchcancel", end as EventListener],
    ["gesturestart", noPageZoom],
    ["gesturechange", noPageZoom],
  ];
  for (const [type, fn] of listeners) el.addEventListener(type, fn, options);
  return () => {
    for (const [type, fn] of listeners) el.removeEventListener(type, fn, options);
  };
}
