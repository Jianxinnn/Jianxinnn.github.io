export type MathJaxWindow = Window & {
  MathJax?: {
    options?: unknown;
    startup?: {
      promise?: Promise<unknown>;
    };
    tex?: unknown;
    typesetPromise?: (elements?: HTMLElement[]) => Promise<unknown>;
  };
};

export function ensureMathJax() {
  const win = window as MathJaxWindow;

  if (win.MathJax?.typesetPromise) {
    return win.MathJax.startup?.promise ?? Promise.resolve();
  }

  const existing = document.querySelector<HTMLScriptElement>("script[data-mathjax]");

  if (existing) {
    return new Promise((resolve) => {
      existing.addEventListener("load", resolve, { once: true });
    });
  }

  win.MathJax = {
    tex: {
      inlineMath: [["$", "$"], ["\\(", "\\)"]],
      displayMath: [["$$", "$$"], ["\\[", "\\]"]],
      processEscapes: true
    },
    options: {
      skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"]
    }
  } as MathJaxWindow["MathJax"];

  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.async = true;
    script.dataset.mathjax = "true";
    script.id = "MathJax-script";
    script.src = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js";
    script.addEventListener("load", resolve, { once: true });
    script.addEventListener("error", reject, { once: true });
    document.head.appendChild(script);
  });
}
