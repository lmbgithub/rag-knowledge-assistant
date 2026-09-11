export type DocStatus = "empty" | "indexing" | "ready" | "failed" | "deleting";

export interface Progress {
  stage: string;
  done: number;
  total: number;
}

export interface DocumentState {
  id: string | null;
  filename: string | null;
  status: DocStatus;
  pages: number;
  chunks: number;
  bytes: number;
  created_at: number | null;
  error: string | null;
  progress: Progress;
}

export interface Citation {
  chunk_id: number;
  page: number;
  text: string;
  score: number;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  partial: boolean;
  created_at: number;
}

export interface ConversationSummary {
  id: string;
  title: string;
  document_id: string | null;
  document_filename: string | null;
  created_at: number;
  updated_at: number;
  message_count: number;
}

export interface Conversation extends ConversationSummary {
  messages: Message[];
}

/** One line of the /chat NDJSON stream. */
export type ChatEvent =
  | { type: "conversation"; conversation: ConversationSummary }
  | { type: "message"; message: Message }
  | { type: "citations"; citations: Citation[] }
  | { type: "token"; text: string }
  | { type: "done"; message: Message; grounded: boolean; document?: string }
  | { type: "error"; status: number; detail: string };
