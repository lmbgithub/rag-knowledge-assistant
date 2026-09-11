"use client";

import * as React from "react";
import { FileText, Trash2, Upload } from "lucide-react";
import { ErrorBanner } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { Dialog } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { formatBytes, progressLabel, progressPercent } from "@/lib/format";
import type { DocumentState } from "@/lib/types";

export function DocumentPanel({
  document,
  busy,
  error,
  onUpload,
  onDelete,
}: {
  document: DocumentState | null;
  busy: boolean;
  error: string | null;
  onUpload: (file: File) => void;
  onDelete: () => void;
}) {
  const [confirming, setConfirming] = React.useState(false);
  const [typed, setTyped] = React.useState("");
  const status = document?.status ?? "empty";

  return (
    <Card>
      <CardHeader className="flex items-center justify-between">
        <span className="text-sm font-semibold">Knowledge base</span>
        {(status === "ready" || status === "deleting") && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              setTyped("");
              setConfirming(true);
            }}
          >
            <Trash2 className="h-4 w-4" aria-hidden />
            Start over
          </Button>
        )}
      </CardHeader>

      <CardBody className="space-y-3">
        {(status === "empty" || status === "failed") && (
          <>
            <p className="text-sm text-muted">
              Upload one PDF. It is parsed, split by page, embedded and indexed before the chat
              opens — there is nothing to answer from until then.
            </p>
            <label className="flex cursor-pointer items-center justify-center gap-2 rounded-md border border-dashed border-line px-4 py-6 text-sm hover:bg-panel">
              <Upload className="h-4 w-4" aria-hidden />
              {busy ? "Uploading…" : "Choose a PDF"}
              <input
                type="file"
                accept="application/pdf,.pdf"
                className="sr-only"
                disabled={busy}
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) onUpload(file);
                  event.target.value = "";
                }}
              />
            </label>
          </>
        )}

        {status === "indexing" && (
          <div className="space-y-2" role="status" aria-live="polite">
            <p className="text-sm">
              Indexing <span className="font-medium">{document?.filename}</span>
            </p>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-line">
              <div
                className="h-full bg-accent transition-[width]"
                style={{ width: `${progressPercent(document!)}%` }}
              />
            </div>
            {/* A count, not a spinner: on CPU this takes minutes and a spinner
                is indistinguishable from a hang. */}
            <p className="text-xs text-muted">{progressLabel(document!)}</p>
          </div>
        )}

        {status === "deleting" && (
          <p className="text-sm text-muted">
            {/* A delete that failed halfway records why. Retrying it is the
                only safe recovery — the half-deleted index must never accept
                new data — so "Start over" stays available above. */}
            {document?.error ? "Deleting failed part-way." : "Deleting…"}
          </p>
        )}

        {status === "ready" && document && (
          <div className="space-y-1 text-sm">
            <p className="flex items-center gap-2 font-medium">
              <FileText className="h-4 w-4 shrink-0" aria-hidden />
              {document.filename}
            </p>
            <p className="text-xs text-muted">
              {document.pages} pages · {document.chunks} passages · {formatBytes(document.bytes)}
            </p>
          </div>
        )}

        {document?.error && <ErrorBanner className="text-xs">{document.error}</ErrorBanner>}

        {error && <ErrorBanner className="text-xs">{error}</ErrorBanner>}
      </CardBody>

      <Dialog open={confirming} onClose={() => setConfirming(false)} title="Delete this document?">
        <p className="mb-3 text-sm text-muted">
          This removes the vector index, every passage and the document record. It cannot be undone.
          Your past conversations are kept.
        </p>
        <p className="mb-2 text-sm">
          Type <span className="font-mono font-medium">{document?.filename}</span> to confirm.
        </p>
        <Input
          value={typed}
          aria-label="Type the filename to confirm"
          onChange={(event) => setTyped(event.target.value)}
        />
        <div className="mt-4 flex justify-end gap-2">
          <Button variant="outline" onClick={() => setConfirming(false)}>
            Cancel
          </Button>
          <Button
            variant="destructive"
            // Typing the name is a deliberate act; an OK button is one misclick.
            disabled={typed !== document?.filename}
            onClick={() => {
              setConfirming(false);
              onDelete();
            }}
          >
            Delete everything
          </Button>
        </div>
      </Dialog>
    </Card>
  );
}
