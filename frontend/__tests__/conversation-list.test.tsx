import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ConversationList } from "@/components/conversation-list";
import type { ConversationSummary } from "@/lib/types";
import { summary } from "@/test-utils/factories";

function list(props: Partial<Parameters<typeof ConversationList>[0]> = {}) {
  const handlers = { onSelect: jest.fn(), onDelete: jest.fn(), onNew: jest.fn() };
  render(
    <ConversationList
      conversations={[summary()]}
      activeId={null}
      currentDocument="paper.pdf"
      {...handlers}
      {...props}
    />,
  );
  return handlers;
}

it("says so when there is nothing yet", () => {
  list({ conversations: [] });
  expect(screen.getByText(/Nothing yet/)).toBeInTheDocument();
});

it("opens a conversation when it is clicked", async () => {
  const { onSelect } = list();
  await userEvent.click(screen.getByText("When did the migration finish?"));
  expect(onSelect).toHaveBeenCalledWith("c1");
});

it("deletes one without opening it", async () => {
  const { onDelete, onSelect } = list();
  await userEvent.click(
    screen.getByRole("button", { name: /Delete conversation When did the migration/ }),
  );
  expect(onDelete).toHaveBeenCalledWith("c1");
  expect(onSelect).not.toHaveBeenCalled();
});

it("names the document a thread was about when it is no longer the current one", () => {
  // Threads outlive the PDF they were about; showing them unlabelled next to
  // a different document's answers would be worse than showing nothing.
  list({ conversations: [summary({ document_filename: "old.pdf" })] });
  expect(screen.getByText(/old\.pdf/)).toBeInTheDocument();
});

it("does not label threads about the current document", () => {
  list();
  expect(screen.queryByText(/paper\.pdf/)).not.toBeInTheDocument();
});

it("starts a new conversation", async () => {
  const { onNew } = list();
  await userEvent.click(screen.getByRole("button", { name: "New conversation" }));
  expect(onNew).toHaveBeenCalled();
});
