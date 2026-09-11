"use client";

import * as React from "react";
import { Chat } from "@/components/chat";
import { ConversationList } from "@/components/conversation-list";
import { DocumentPanel } from "@/components/document-panel";
import * as api from "@/lib/api";
import { DRAFT_ID, errorMessage, settleMessages } from "@/lib/format";
import type { ConversationSummary, DocumentState, Message } from "@/lib/types";

const POLL_MS = 1000;

let placeholderSeq = 0;

/**
 * A locally-created message, awaiting the server's version.
 *
 * The id is unique per call. A fixed `"asked"` id was the first version, and
 * it collided the moment a question failed before the server answered: the
 * stale bubble survived, the next question pushed a second one with the same
 * id, React saw duplicate keys, and the `message` event replaced *both* with
 * one server message — rendering the new question twice and losing the old.
 */
const placeholder = (role: Message["role"], content: string): Message => ({
  id: `local-${(placeholderSeq += 1)}`,
  role,
  content,
  citations: [],
  partial: false,
  created_at: Date.now() / 1000,
});

export default function Page() {
  const [document, setDocument] = React.useState<DocumentState | null>(null);
  const [conversations, setConversations] = React.useState<ConversationSummary[]>([]);
  const [activeId, setActiveId] = React.useState<string | null>(null);
  const [messages, setMessages] = React.useState<Message[]>([]);
  const [streaming, setStreaming] = React.useState(false);
  const [busy, setBusy] = React.useState(false);
  const [documentError, setDocumentError] = React.useState<string | null>(null);
  const [chatError, setChatError] = React.useState<string | null>(null);

  const refreshConversations = React.useCallback(async () => {
    try {
      setConversations(await api.listConversations());
    } catch {
      // The list is a convenience; a failure here must not blank the chat.
    }
  }, []);

  React.useEffect(() => {
    void api
      .getDocument()
      .then(setDocument)
      .catch(() => setDocument(null));
    void refreshConversations();
  }, [refreshConversations]);

  // Poll only while indexing is actually running. Polling a ready document
  // forever is a request per second for a value that cannot change.
  React.useEffect(() => {
    if (document?.status !== "indexing" && document?.status !== "deleting") return;
    const timer = setInterval(() => {
      void api
        .getDocument()
        .then(setDocument)
        .catch(() => undefined);
    }, POLL_MS);
    return () => clearInterval(timer);
  }, [document?.status]);

  const upload = async (file: File) => {
    setBusy(true);
    setDocumentError(null);
    try {
      setDocument(await api.uploadDocument(file));
    } catch (error) {
      setDocumentError(errorMessage(error));
    } finally {
      setBusy(false);
    }
  };

  const remove = async () => {
    if (!document?.filename) return;
    setDocumentError(null);
    try {
      setDocument(await api.deleteDocument(document.filename));
      setActiveId(null);
      setMessages([]);
    } catch (error) {
      setDocumentError(errorMessage(error));
      // The record is very likely `deleting` now — a delete that raised
      // part-way leaves it there. Without this the panel keeps showing a
      // healthy document and the composer stays enabled until a reload,
      // because the polling effect keys off a status that never changed.
      void api
        .getDocument()
        .then(setDocument)
        .catch(() => undefined);
    }
  };

  const ask = async (question: string) => {
    setChatError(null);
    setStreaming(true);
    const asked = placeholder("user", question);
    const draft: Message = { ...placeholder("assistant", ""), id: DRAFT_ID };
    setMessages((current) => [...current, asked, draft]);

    // The draft is always the last element, so it is patched in place rather
    // than by scanning and rebuilding the array on every token.
    const patchDraft = (patch: (draft: Message) => Message) =>
      setMessages((current) => {
        const next = current.slice();
        const last = next.length - 1;
        if (last >= 0 && next[last].id === DRAFT_ID) next[last] = patch(next[last]);
        return next;
      });

    // Both placeholders go: leaving the question behind is what produced the
    // duplicate-key collision above.
    const dropPlaceholders = () =>
      setMessages((current) => current.filter((m) => m.id !== DRAFT_ID && m.id !== asked.id));

    try {
      for await (const event of api.streamChat(question, activeId)) {
        switch (event.type) {
          case "conversation":
            setActiveId(event.conversation.id);
            break;
          case "message":
            // The server's copy of the question, with its real id. The
            // optimistic one is a placeholder sharing a fixed id, which would
            // collide as a React key the moment a second question was asked.
            setMessages((current) => current.map((m) => (m.id === asked.id ? event.message : m)));
            break;
          case "citations":
            patchDraft((draft) => ({ ...draft, citations: event.citations }));
            break;
          case "token":
            patchDraft((draft) => ({ ...draft, content: draft.content + event.text }));
            break;
          case "done":
            // The server's message is authoritative: a refusal token that was
            // streamed verbatim is rewritten in it.
            setMessages((current) => settleMessages(current, event.message));
            break;
          case "error":
            setChatError(event.detail);
            dropPlaceholders();
            break;
        }
      }
    } catch (error) {
      setChatError(errorMessage(error));
      dropPlaceholders();
    } finally {
      setStreaming(false);
      void refreshConversations();
    }
  };

  const openConversation = async (id: string) => {
    setChatError(null);
    try {
      const conversation = await api.getConversation(id);
      setActiveId(id);
      setMessages(conversation.messages);
    } catch (error) {
      setChatError(errorMessage(error));
    }
  };

  const removeConversation = async (id: string) => {
    await api.deleteConversation(id).catch(() => undefined);
    if (id === activeId) {
      setActiveId(null);
      setMessages([]);
    }
    void refreshConversations();
  };

  return (
    <main className="flex h-full">
      <aside className="flex w-80 shrink-0 flex-col border-r border-line">
        <div className="p-3">
          <DocumentPanel
            document={document}
            busy={busy}
            error={documentError}
            onUpload={upload}
            onDelete={remove}
          />
        </div>
        <div className="min-h-0 flex-1 border-t border-line">
          <ConversationList
            conversations={conversations}
            activeId={activeId}
            currentDocument={document?.filename ?? null}
            onSelect={openConversation}
            onDelete={removeConversation}
            onNew={() => {
              setActiveId(null);
              setMessages([]);
              setChatError(null);
            }}
          />
        </div>
      </aside>

      <section className="min-w-0 flex-1">
        <Chat
          document={document}
          messages={messages}
          streaming={streaming}
          error={chatError}
          onAsk={ask}
        />
      </section>
    </main>
  );
}
