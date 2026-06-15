// MathJax config expected by pymdownx.arithmatex (generic mode).
//
// CRITICAL: only set window.MathJax if it isn't already defined. Material's
// instant-nav re-executes <body> scripts on every page swap, which means
// this file runs again — and unconditionally assigning
// `window.MathJax = {tex, options}` would overwrite the live library
// (with its typesetPromise / startup / etc.) with a stub config object,
// silently breaking every subsequent re-typeset. The math on the first
// page renders because MathJax completes its initial typeset *before*
// the post-load instant-nav clobber; click any link and the new page's
// arithmatex spans stay as raw `\(...\)` text.
if (!window.MathJax) {
  window.MathJax = {
    tex: {
      inlineMath: [["\\(", "\\)"]],
      displayMath: [["\\[", "\\]"]],
      processEscapes: true,
      processEnvironments: true
    },
    options: {
      ignoreHtmlClass: ".*|",
      processHtmlClass: "arithmatex"
    }
  };
}

// Belt-and-suspenders typeset boot. typesetPromise is idempotent over
// already-rendered output, so calling it twice on the same page is
// harmless. The DOMContentLoaded path covers direct loads where the
// MathJax library hasn't quite finished setup by the time document$
// would normally have emitted; document$.subscribe covers Material's
// instant-nav swaps where DOMContentLoaded does NOT re-fire.
function typesetMath() {
  if (window.MathJax && MathJax.typesetPromise) {
    MathJax.typesetPromise();
  }
}
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", typesetMath);
} else {
  typesetMath();
}
if (typeof document$ !== "undefined" && document$.subscribe) {
  document$.subscribe(typesetMath);
}
