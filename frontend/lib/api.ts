/**
 * The API client.
 *
 * `streamChat` is the only interesting part: it reads NDJSON off a
 * `ReadableStream` and yields one parsed event per line. The subtlety is that
 * a chunk boundary lands anywhere, including mid-object, so the tail of a
 * chunk is carried into the next one instead of being parsed and dropped.
 */
import type { ChatEvent, Conversation, ConversationSummary, DocumentState } from "./types";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/**
 * The one place a failed response becomes an ApiError.
 *
 * `streamChat` cannot use `request` — it needs the raw body — but it needs
 * exactly this, and used to carry its own copy of it.
 */
async function throwIfNotOk(response: Response): Promise<void> {
  if (response.ok) return;
  let detail = response.statusText;
  try {
    detail = (await response.json()).detail ?? detail;
  } catch {
    // A non-JSON error body is still an error; the status carries the meaning.
  }
  throw new ApiError(response.status, detail);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, init);
  await throwIfNotOk(response);
  return (await response.json()) as T;
}

export const getDocument = () => request<DocumentState>("/document");

export const listConversations = () =>
  request<{ conversations: ConversationSummary[] }>("/conversations").then((r) => r.conversations);

export const getConversation = (id: string) => request<Conversation>(`/conversations/${id}`);

export const deleteConversation = (id: string) =>
  request<{ deleted: string }>(`/conversations/${id}`, { method: "DELETE" });

export function uploadDocument(file: File): Promise<DocumentState> {
  const body = new FormData();
  body.append("file", file);
  return request<DocumentState>("/document", { method: "POST", body });
}

/** `confirm` must equal the filename; the server enforces it too. */
export const deleteDocument = (confirm: string) =>
  request<DocumentState>(`/document?confirm=${encodeURIComponent(confirm)}`, { method: "DELETE" });

/** Split NDJSON into events, carrying an incomplete tail across chunks. */
export function ndjson(): (chunk: string, done?: boolean) => ChatEvent[] {
  let buffer = "";
  return (chunk: string, done = false) => {
    buffer += chunk;
    const lines = buffer.split("\n");
    buffer = done ? "" : (lines.pop() ?? "");
    if (done && lines.length && lines[lines.length - 1] === "") lines.pop();
    const events: ChatEvent[] = [];
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        events.push(JSON.parse(trimmed) as ChatEvent);
      } catch {
        // A frame we cannot parse is dropped rather than aborting the stream:
        // the answer already on screen is worth more than the parse error.
      }
    }
    return events;
  };
}

export async function* streamChat(
  question: string,
  conversationId: string | null,
  signal?: AbortSignal,
): AsyncGenerator<ChatEvent> {
  const response = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, conversation_id: conversationId }),
    signal,
  });

  await throwIfNotOk(response);
  if (!response.body) throw new ApiError(500, "The response had no body.");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const parse = ndjson();
  for (;;) {
    const { value, done } = await reader.read();
    const events = parse(value ? decoder.decode(value, { stream: !done }) : "", done);
    for (const event of events) yield event;
    if (done) return;
  }
}
