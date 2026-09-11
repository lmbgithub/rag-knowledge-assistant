import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { DocumentPanel } from "@/components/document-panel";
import type { DocumentState } from "@/lib/types";
import { doc } from "@/test-utils/factories";

const noop = () => undefined;

function panel(
  document: DocumentState | null,
  props: Partial<Parameters<typeof DocumentPanel>[0]> = {},
) {
  return render(
    <DocumentPanel
      document={document}
      busy={false}
      error={null}
      onUpload={noop}
      onDelete={noop}
      {...props}
    />,
  );
}

it("offers an upload box and nothing else when empty", () => {
  panel(doc({ status: "empty" }));
  expect(screen.getByText("Choose a PDF")).toBeInTheDocument();
  expect(screen.queryByText("Start over")).not.toBeInTheDocument();
});

it("shows a real count while indexing, not a spinner", () => {
  panel(doc({ status: "indexing", progress: { stage: "embedding", done: 4, total: 12 } }));
  expect(screen.getByText("embedding 4/12")).toBeInTheDocument();
});

it("shows the document's real counts when ready", () => {
  panel(doc());
  expect(screen.getByText(/2 pages · 9 passages · 2 KB/)).toBeInTheDocument();
});

it("surfaces the reason a document failed", () => {
  panel(doc({ status: "failed", error: "scan.pdf has no text layer — it looks like a scan." }));
  expect(screen.getByText(/no text layer/)).toBeInTheDocument();
});

it("lets a failed document be replaced", () => {
  panel(doc({ status: "failed", error: "boom" }));
  expect(screen.getByText("Choose a PDF")).toBeInTheDocument();
});

it("requires the filename to be typed before deleting", async () => {
  const onDelete = jest.fn();
  panel(doc(), { onDelete });
  await userEvent.click(screen.getByText("Start over"));

  const button = screen.getByRole("button", { name: "Delete everything" });
  expect(button).toBeDisabled();

  await userEvent.type(screen.getByLabelText("Type the filename to confirm"), "wrong.pdf");
  expect(button).toBeDisabled();

  await userEvent.clear(screen.getByLabelText("Type the filename to confirm"));
  await userEvent.type(screen.getByLabelText("Type the filename to confirm"), "paper.pdf");
  expect(button).toBeEnabled();

  await userEvent.click(button);
  expect(onDelete).toHaveBeenCalledTimes(1);
});

it("says conversations are kept, because deleting a PDF must not destroy them", async () => {
  panel(doc());
  await userEvent.click(screen.getByText("Start over"));
  expect(screen.getByText(/past conversations are kept/i)).toBeInTheDocument();
});

it("closes the confirmation on escape without deleting", async () => {
  const onDelete = jest.fn();
  panel(doc(), { onDelete });
  await userEvent.click(screen.getByText("Start over"));
  fireEvent.keyDown(window.document, { key: "Escape" });
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(onDelete).not.toHaveBeenCalled();
});

it("reports an upload error", () => {
  panel(doc({ status: "empty" }), { error: "paper.pdf is not a PDF" });
  expect(screen.getByText("paper.pdf is not a PDF")).toBeInTheDocument();
});

it("offers the delete again when one failed part-way", async () => {
  // `deleting` used to have no exit: the panel hid "Start over" for every
  // status but ready, so the only safe recovery — retrying the delete — was
  // unreachable, and the frontend polled forever with nothing shown.
  const onDelete = jest.fn();
  panel(doc({ status: "deleting", error: "cannot reach MongoDB" }), { onDelete });
  expect(screen.getByText("Deleting failed part-way.")).toBeInTheDocument();
  expect(screen.getByText("cannot reach MongoDB")).toBeInTheDocument();
  expect(screen.getByText("Start over")).toBeInTheDocument();
});

it("shows no error while a delete is still running", () => {
  panel(doc({ status: "deleting" }));
  expect(screen.getByText("Deleting…")).toBeInTheDocument();
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
});

it("does not offer an upload box while deleting", () => {
  panel(doc({ status: "deleting", error: "boom" }));
  expect(screen.queryByText("Choose a PDF")).not.toBeInTheDocument();
});
