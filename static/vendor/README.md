# Vendored browser libraries

The two browser libraries of the architecture page (`titlestd/explorer.py`), copied
unchanged. Design 2.8 allows these two and no others, and only the owner adds, edits or
upgrades a file here. `tests/unit/test_hygiene_vendor.py` checks every file against
`SHA256SUMS`.

| File | Library | Version | Source | Bytes | Licence |
|---|---|---|---|---|---|
| `htmx.min.js` | htmx | 2.0.11 | https://cdn.jsdelivr.net/npm/htmx.org@2.0.11/dist/htmx.min.js | 52182 | `LICENSE-htmx.txt` (0BSD) |
| `mermaid.min.js` | Mermaid | 11.17.2 | https://cdn.jsdelivr.net/npm/mermaid@11.17.2/dist/mermaid.min.js | 3572661 | `LICENSE-mermaid.txt` (MIT) |

Each licence file comes from the same release: `LICENSE` beside the `dist/` folder of
https://cdn.jsdelivr.net/npm/htmx.org@2.0.11/ and https://cdn.jsdelivr.net/npm/mermaid@11.17.2/.
`mermaid.min.js` carries the notices of the libraries bundled in it, kept as they are.

Fetched on 2026-10-21 and checked against a second source before they were copied here:
each package's tarball from the npm registry matched the registry's published sha512, and
the four files are byte-identical to the ones inside those tarballs.

## How the page loads them

- **htmx.** The page head carries
  `<meta name="htmx-config" content='{"includeIndicatorStyles":false,"globalViewTransitions":false,"defaultSettleDelay":0,"allowScriptTags":false,"allowEval":false,"historyCacheSize":0}'>`.
  Never use `transition:true` in `hx-swap`, and write no CSS transitions or animations.
- **Mermaid.** `static/app.js`, never an inline script, calls
  `mermaid.initialize({startOnLoad: false, securityLevel: "strict"})`, then, from
  `htmx.onLoad`, `mermaid.run` on every new `pre.mermaid`, the loaded element itself
  included.
- **No record text in a flowchart.** `strict` sanitizes the HTML of a label but does not
  encode it, and `run()` decodes entities first, so every label comes from a constant in the
  code.
- **The Content-Security-Policy the page sends:** `default-src 'none'; script-src 'self';
  style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; base-uri 'none';
  form-action 'none'; frame-ancestors 'none'`. Mermaid creates `<style>` elements, so styles
  need `'unsafe-inline'`. It calls neither `eval` nor `new Function`, so scripts stay `'self'`.
