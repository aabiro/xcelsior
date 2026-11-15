import "@testing-library/jest-dom/vitest";

// jsdom doesn't implement scrollIntoView
if (typeof Element !== "undefined") {
  Element.prototype.scrollIntoView = () => {};
}
