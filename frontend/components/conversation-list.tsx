"use client";

import { MessageSquare, Plus, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { relativeTime } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { ConversationSummary } from "@/lib/types";

export function ConversationList({
  conversations,
  activeId,
  currentDocument,
  onSelect,
  onDelete,
  onNew,
}: {
  conversations: ConversationSummary[];
  activeId: string | null;
  currentDocument: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onNew: () => void;
}) {
  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between px-3 py-3">
        <span className="text-sm font-semibold">Conversations</span>
        <Button variant="ghost" size="icon" aria-label="New conversation" onClick={onNew}>
          <Plus className="h-4 w-4" aria-hidden />
        </Button>
      </div>

      <ul className="flex-1 space-y-1 overflow-y-auto px-2 pb-3">
        {conversations.length === 0 && (
          <li className="px-2 py-4 text-xs text-muted">Nothing yet. Ask a question.</li>
        )}
        {conversations.map((conversation) => {
          const orphaned =
            currentDocument !== null && conversation.document_filename !== currentDocument;
          return (
            <li key={conversation.id}>
              <div
                className={cn(
                  "group flex items-start gap-2 rounded-md px-2 py-2 text-left text-sm hover:bg-panel",
                  conversation.id === activeId && "bg-panel",
                )}
              >
                <button
                  className="flex min-w-0 flex-1 items-start gap-2 text-left"
                  onClick={() => onSelect(conversation.id)}
                >
                  <MessageSquare className="mt-0.5 h-4 w-4 shrink-0 text-muted" aria-hidden />
                  <span className="min-w-0">
                    <span className="block truncate">{conversation.title}</span>
                    <span className="block text-xs text-muted">
                      {relativeTime(conversation.updated_at)} · {conversation.message_count}{" "}
                      messages
                      {/* Threads outlive the document they were about; saying
                          so beats showing citations from a different PDF. */}
                      {orphaned && ` · ${conversation.document_filename ?? "deleted"}`}
                    </span>
                  </span>
                </button>
                <button
                  aria-label={`Delete conversation ${conversation.title}`}
                  className="opacity-0 transition-opacity group-hover:opacity-100"
                  onClick={() => onDelete(conversation.id)}
                >
                  <Trash2 className="h-4 w-4 text-muted hover:text-red-500" aria-hidden />
                </button>
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
