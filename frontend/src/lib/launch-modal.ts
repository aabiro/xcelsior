"use client";

/**
 * Lightweight global launch-modal bus.
 *
 * Any component can call `openLaunchModal()` to pop the Launch Instance modal
 * in place — no route change, no detour through the instances list, and never a
 * jump to the instance detail page. A single <GlobalLaunchModal> mounted in the
 * dashboard shell listens for the event and renders the modal.
 */

export const LAUNCH_MODAL_EVENT = "xcelsior:open-launch-modal";

export interface LaunchModalOptions {
  /** Pre-select a GPU model. */
  gpu?: string;
  /** Pre-select a pricing mode. */
  mode?: "spot" | "on_demand";
  /** Pre-attach volumes. */
  volumeIds?: string[];
}

export function openLaunchModal(options: LaunchModalOptions = {}): void {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new CustomEvent<LaunchModalOptions>(LAUNCH_MODAL_EVENT, { detail: options }));
}
