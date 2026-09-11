import {
  DRAFT_ID,
  canAsk,
  formatBytes,
  progressLabel,
  progressPercent,
  relativeTime,
  settleMessages,
} from "@/lib/format";
import type { DocumentState, Message } from "@/lib/types";
import { doc, message } from "@/test-utils/factories";

describe("formatBytes", () => {
  it.each([
    [12, "12 B"],
    [2048, "2 KB"],
    [5 * 1024 * 1024, "5.0 MB"],
  ])("formats %i", (bytes, expected) => {
    expect(formatBytes(bytes)).toBe(expected);
  });
});

describe("progress", () => {
  it("shows a count, not a spinner", () => {
    expect(progressLabel(doc({ progress: { stage: "embedding", done: 3, total: 9 } }))).toBe(
      "embedding 3/9",
    );
  });

  it("omits the count when there is nothing to count yet", () => {
    expect(progressLabel(doc({ progress: { stage: "reading the PDF", done: 0, total: 0 } }))).toBe(
      "reading the PDF",
    );
  });

  it("says starting rather than showing an empty label", () => {
    expect(progressLabel(doc())).toBe("starting…");
  });

  it("never divides by zero", () => {
    expect(progressPercent(doc())).toBe(0);
  });

  it("clamps above 100", () => {
    expect(progressPercent(doc({ progress: { stage: "e", done: 20, total: 9 } }))).toBe(100);
  });
});

describe("canAsk", () => {
  it("is false with no document", () => {
    expect(canAsk(null)).toBe(false);
  });

  it.each(["empty", "indexing", "failed", "deleting"] as const)("is false while %s", (status) => {
    expect(canAsk(doc({ status }))).toBe(false);
  });

  it("is true only when ready", () => {
    expect(canAsk(doc({ status: "ready" }))).toBe(true);
  });
});

describe("relativeTime", () => {
  it.each([
    [10, "just now"],
    [120, "2m ago"],
    [7200, "2h ago"],
    [172800, "2d ago"],
  ])("renders %i seconds ago", (ago, expected) => {
    const now = 1_000_000;
    expect(relativeTime(now - ago, now)).toBe(expected);
  });

  it("does not render a future timestamp as negative", () => {
    expect(relativeTime(2000, 1000)).toBe("just now");
  });
});

describe("settleMessages", () => {
  it("replaces the streamed draft with the server's version", () => {
    const streamed = [
      message({ id: "asked", role: "user" }),
      message({ id: "draft", content: "NOT_IN_DOCUMENT" }),
    ];
    const final = message({ id: "real", content: "I could not find anything…" });
    expect(settleMessages(streamed, final)).toEqual([streamed[0], final]);
  });

  it("keeps earlier messages untouched", () => {
    const history = [message({ id: "old" })];
    expect(settleMessages(history, message({ id: "new" })).map((m) => m.id)).toEqual([
      "old",
      "new",
    ]);
  });
});

describe("draft settling", () => {
  it("removes the draft by its own id, leaving the question in place", () => {
    const question = message({ id: "local-1", role: "user", content: "when?" });
    const draft = message({ id: DRAFT_ID, content: "NOT_IN_DOC" });
    const final = message({ id: "server-1", content: "In March." });
    expect(settleMessages([question, draft], final).map((m) => m.id)).toEqual([
      "local-1",
      "server-1",
    ]);
  });

  it("is a no-op on the draft when there is none", () => {
    const history = [message({ id: "a" })];
    expect(settleMessages(history, message({ id: "b" })).map((m) => m.id)).toEqual(["a", "b"]);
  });
});
