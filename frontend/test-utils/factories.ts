/**
 * Shared test fixtures.
 *
 * These three builders were copied byte-for-byte into three test files, so a
 * field added to `DocumentState` or `Message` in lib/types.ts needed the same
 * edit three times and TypeScript only complained once per file.
 */
import type { ConversationSummary, DocumentState, Message } from "@/lib/types";

export const doc = (over: Partial<DocumentState> = {}): DocumentState => ({
  id: "d",
  filename: "paper.pdf",
  status: "ready",
  pages: 2,
  chunks: 9,
  bytes: 2048,
  created_at: 0,
  error: null,
  progress: { stage: "", done: 0, total: 0 },
  ...over,
});

export const message = (over: Partial<Message> = {}): Message => ({
  id: "m",
  role: "assistant",
  content: "The migration completed in March.",
  citations: [],
  partial: false,
  created_at: 0,
  ...over,
});

export const summary = (over: Partial<ConversationSummary> = {}): ConversationSummary => ({
  id: "c1",
  title: "When did the migration finish?",
  document_id: "d",
  document_filename: "paper.pdf",
  created_at: Date.now() / 1000,
  updated_at: Date.now() / 1000,
  message_count: 2,
  ...over,
});
