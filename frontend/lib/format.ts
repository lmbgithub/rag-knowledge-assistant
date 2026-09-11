import type { DocumentState, Message } from "./types";

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** The one-line status a person reads while indexing runs. */
export function progressLabel(document: DocumentState): string {
  const { stage, done, total } = document.progress;
  if (!stage) return "starting…";
  if (total > 0) return `${stage} ${done}/${total}`;
  return stage;
}

export function progressPercent(document: DocumentState): number {
  const { done, total } = document.progress;
  if (total <= 0) return 0;
  return Math.min(100, Math.round((done / total) * 100));
}

/**
 * Whether the composer accepts input.
 *
 * The whole point of the state machine: with no ready document there is
 * nothing to ground an answer in, so the input is disabled rather than
 * accepting a question the server will refuse.
 */
export const canAsk = (document: DocumentState | null): boolean => document?.status === "ready";

export function relativeTime(seconds: number, now: number = Date.now() / 1000): string {
  const delta = Math.max(0, now - seconds);
  if (delta < 60) return "just now";
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
  return `${Math.floor(delta / 86400)}d ago`;
}

/**
 * Replace the streamed draft with the server's authoritative version.
 *
 * The tokens on screen are what the model emitted; `done.message.content` is
 * what was stored, and the two differ when the refusal token was rewritten.
 * Rendering the accumulation would leave `NOT_IN_DOCUMENT` on screen.
 */
export function settleMessages(messages: Message[], final: Message): Message[] {
  return [...messages.filter((m) => m.id !== DRAFT_ID), final];
}

/** The id the in-flight assistant message carries until the server settles it. */
export const DRAFT_ID = "draft";

/**
 * The app's single error-to-string rule.
 *
 * It was written inline four times in page.tsx, so changing it — to surface an
 * ApiError's status, say — meant four edits with no single test target.
 */
export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
