"use client";

import * as React from "react";
import { SendHorizonal } from "lucide-react";
import { ErrorBanner } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/input";
import { canAsk } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { Citation, DocumentState, Message } from "@/lib/types";

function Citations({ citations }: { citations: Citation[] }) {
  if (citations.length === 0) return null;
  return (
    <details className="mt-2 rounded-md border border-line bg-surface/60 px-3 py-2 text-xs">
      <summary className="cursor-pointer text-muted">
        {citations.length} passage{citations.length === 1 ? "" : "s"} used
      </summary>
      <ol className="mt-2 space-y-2">
        {citations.map((citation, i) => (
          <li key={citation.chunk_id}>
            <span className="font-medium">
              [{i + 1}] page {citation.page}
            </span>
            <span className="text-muted"> · similarity {citation.score.toFixed(3)}</span>
            <p className="mt-1 whitespace-pre-wrap text-muted">{citation.text}</p>
          </li>
        ))}
      </ol>
    </details>
  );
}

const Bubble = React.memo(function Bubble({ message }: { message: Message }) {
  const mine = message.role === "user";
  return (
    <div className={cn("flex", mine ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "max-w-[46rem] rounded-lg px-4 py-3 text-sm",
          mine ? "bg-accent text-white" : "bg-panel",
        )}
      >
        <p className="whitespace-pre-wrap">{message.content}</p>
        {message.partial && (
          // An interrupted stream is labelled, never silently completed.
          <p className="mt-1 text-xs opacity-70">interrupted</p>
        )}
        {!mine && <Citations citations={message.citations} />}
      </div>
    </div>
  );
});

export function Chat({
  document,
  messages,
  streaming,
  error,
  onAsk,
}: {
  document: DocumentState | null;
  messages: Message[];
  streaming: boolean;
  error: string | null;
  onAsk: (question: string) => void;
}) {
  const [question, setQuestion] = React.useState("");
  const bottom = React.useRef<HTMLDivElement>(null);
  const ready = canAsk(document);

  React.useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streaming]);

  const submit = () => {
    const text = question.trim();
    if (!text || streaming || !ready) return;
    setQuestion("");
    onAsk(text);
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 space-y-4 overflow-y-auto px-6 py-6">
        {messages.length === 0 && (
          <p className="mx-auto max-w-lg pt-16 text-center text-sm text-muted">
            {ready
              ? `Ask anything about ${document?.filename}. Every answer shows the passages it came from.`
              : "Upload a PDF to start. Until it is indexed there is nothing to answer from."}
          </p>
        )}
        {messages.map((message) => (
          <Bubble key={message.id} message={message} />
        ))}
        {error && <ErrorBanner className="text-sm">{error}</ErrorBanner>}
        <div ref={bottom} />
      </div>

      <div className="border-t border-line px-6 py-4">
        <div className="flex items-end gap-2">
          <Textarea
            rows={2}
            value={question}
            disabled={!ready || streaming}
            aria-label="Question"
            placeholder={ready ? "Ask about this document…" : "Upload a PDF first"}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                submit();
              }
            }}
          />
          <Button
            aria-label="Send"
            disabled={!ready || streaming || question.trim().length === 0}
            onClick={submit}
          >
            <SendHorizonal className="h-4 w-4" aria-hidden />
          </Button>
        </div>
        <p className="mt-2 text-xs text-muted">
          Answers come only from the indexed PDF. If nothing in it matches, it says so.
        </p>
      </div>
    </div>
  );
}
