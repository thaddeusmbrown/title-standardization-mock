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
