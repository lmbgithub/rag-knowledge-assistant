import "@testing-library/jest-dom";

// jsdom does not implement scrollIntoView, and the chat pane calls it on every
// new message. Stubbing it here keeps the component honest — the alternative
// is guarding the call in production code for a test's benefit.
window.HTMLElement.prototype.scrollIntoView = jest.fn();
