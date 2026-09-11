import { ndjson } from "@/lib/api";

describe("ndjson", () => {
  it("parses one event per line", () => {
    const parse = ndjson();
    const events = parse('{"type":"token","text":"a"}\n{"type":"token","text":"b"}\n');
    expect(events.map((e) => e.type)).toEqual(["token", "token"]);
  });

  it("carries an incomplete object across chunk boundaries", () => {
    // The failure this exists to prevent: a chunk boundary mid-object drops
    // the event entirely and the answer loses a word with no error anywhere.
    const parse = ndjson();
    expect(parse('{"type":"token","te')).toEqual([]);
    expect(parse('xt":"hello"}\n')).toEqual([{ type: "token", text: "hello" }]);
  });

  it("holds back a complete line with no newline until the stream ends", () => {
    const parse = ndjson();
    expect(parse('{"type":"done","message":{},"grounded":true}')).toEqual([]);
    expect(parse("", true)).toHaveLength(1);
  });

  it("ignores blank lines", () => {
    expect(ndjson()('\n\n{"type":"token","text":"a"}\n')).toHaveLength(1);
  });

  it("drops an unparseable frame without losing the rest", () => {
    const events = ndjson()('{ not json\n{"type":"token","text":"a"}\n');
    expect(events).toEqual([{ type: "token", text: "a" }]);
  });

  it("returns nothing for an empty stream", () => {
    expect(ndjson()("", true)).toEqual([]);
  });
});
