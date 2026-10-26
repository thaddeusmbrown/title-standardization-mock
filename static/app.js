// The architecture page (titlestd/explorer.py). Mermaid draws the flowchart of
// every part of the page htmx loads, the first page included, configured as
// static/vendor/README.md says: no start on load, the strict security level.
"use strict";

mermaid.initialize({startOnLoad: false, securityLevel: "strict"});

htmx.onLoad(function (element) {
  if (!(element instanceof Element)) {
    return;
  }
  const charts = Array.from(element.querySelectorAll("pre.mermaid"));
  if (element.matches("pre.mermaid")) {
    charts.unshift(element);
  }
  if (charts.length > 0) {
    mermaid.run({nodes: charts});
  }
});

// htmx swaps no error into the page: a 404, the 503 of a state that is gone or
// not a free run's, or a page that stopped. The address is loaded in full
// instead, so its answer is seen.
for (const name of ["htmx:responseError", "htmx:sendError"]) {
  document.addEventListener(name, function (event) {
    window.location.assign(event.detail.pathInfo.requestPath);
  });
}
document.addEventListener("htmx:historyCacheMissLoadError", function () {
  window.location.reload();
});
// A history restore that cannot reach the page fires no htmx event of its own.
document.addEventListener("htmx:historyCacheMiss", function (event) {
  event.detail.xhr.addEventListener("error", function () {
    window.location.reload();
  });
});
