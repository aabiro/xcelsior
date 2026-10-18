/** Keep docs navigation in the same tab (logo + external navbar links). */
(function () {
  function sameTab(el) {
    if (!el || el.tagName !== "A") return;
    const href = el.getAttribute("href");
    if (!href || href.startsWith("#") || href.startsWith("mailto:")) return;
    el.setAttribute("target", "_self");
    el.removeAttribute("rel");
  }

  function patch(root) {
    root.querySelectorAll("a[href]").forEach(sameTab);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => patch(document));
  } else {
    patch(document);
  }

  const observer = new MutationObserver((mutations) => {
    for (const m of mutations) {
      for (const node of m.addedNodes) {
        if (node.nodeType !== Node.ELEMENT_NODE) continue;
        sameTab(node);
        if (node.querySelectorAll) patch(node);
      }
    }
  });
  observer.observe(document.documentElement, { childList: true, subtree: true });
})();