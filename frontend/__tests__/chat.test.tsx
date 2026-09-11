import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Chat } from "@/components/chat";
import type { DocumentState, Message } from "@/lib/types";
import { doc, message } from "@/test-utils/factories";

function chat(props: Partial<Parameters<typeof Chat>[0]> = {}) {
  const onAsk = jest.fn();
  render(
    <Chat document={doc()} messages={[]} streaming={false} error={null} onAsk={onAsk} {...props} />,
  );
  return onAsk;
}

it("disables the composer until a document is ready", () => {
  chat({ document: doc({ status: "indexing" }) });
  expect(screen.getByLabelText("Question")).toBeDisabled();
  expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
});

it("disables the composer with no document at all", () => {
  chat({ document: null });
  expect(screen.getByLabelText("Question")).toBeDisabled();
});

it("enables it once ready", () => {
  chat();
  expect(screen.getByLabelText("Question")).toBeEnabled();
});

it("will not send an empty or whitespace question", async () => {
  const onAsk = chat();
  await userEvent.type(screen.getByLabelText("Question"), "   ");
  expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
  expect(onAsk).not.toHaveBeenCalled();
});

it("sends on enter and clears the box", async () => {
  const onAsk = chat();
  const box = screen.getByLabelText("Question");
  await userEvent.type(box, "when did it finish?{Enter}");
  expect(onAsk).toHaveBeenCalledWith("when did it finish?");
  expect(box).toHaveValue("");
});

it("shift+enter is a newline, not a send", async () => {
  const onAsk = chat();
  await userEvent.type(
    screen.getByLabelText("Question"),
    "line one{Shift>}{Enter}{/Shift}line two",
  );
  expect(onAsk).not.toHaveBeenCalled();
});

it("will not send a second question while one is streaming", async () => {
  const onAsk = chat({ streaming: true });
  await userEvent.type(screen.getByLabelText("Question"), "again{Enter}");
  expect(onAsk).not.toHaveBeenCalled();
});

it("renders the passages an answer came from", async () => {
  chat({
    messages: [
      message({
        citations: [{ chunk_id: 1, page: 3, text: "It finished in March.", score: 0.812 }],
      }),
    ],
  });
  await userEvent.click(screen.getByText("1 passage used"));
  expect(screen.getByText("[1] page 3")).toBeInTheDocument();
  expect(screen.getByText(/similarity 0.812/)).toBeInTheDocument();
});

it("shows no citation block when there are none", () => {
  chat({ messages: [message()] });
  expect(screen.queryByText(/passages? used/)).not.toBeInTheDocument();
});

it("labels an interrupted answer rather than passing it off as complete", () => {
  chat({ messages: [message({ content: "The migration compl", partial: true })] });
  expect(screen.getByText("interrupted")).toBeInTheDocument();
});

it("shows an error without discarding the conversation", () => {
  chat({ messages: [message()], error: "cannot reach ollama" });
  expect(screen.getByText("cannot reach ollama")).toBeInTheDocument();
  expect(screen.getByText(/completed in March/)).toBeInTheDocument();
});

it("tells the user to upload before anything else", () => {
  chat({ document: null });
  expect(screen.getByText(/Upload a PDF to start/)).toBeInTheDocument();
});
