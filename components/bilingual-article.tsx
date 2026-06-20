"use client";

import { Columns2, Languages } from "lucide-react";
import { useEffect, useRef, useState } from "react";

export type BilingualSegment = {
  en: string;
  id?: string;
  kind?: "heading" | "paragraph" | "html";
  level?: 2 | 3 | 4;
  zh: string;
};

type BilingualArticleProps = {
  segments: BilingualSegment[];
};

type BilingualMode = "zh" | "both" | "en";
type MathJaxWindow = Window & {
  MathJax?: {
    options?: unknown;
    startup?: {
      promise?: Promise<unknown>;
    };
    tex?: unknown;
    typesetPromise?: (elements?: HTMLElement[]) => Promise<unknown>;
  };
};

function ensureMathJax() {
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

function SegmentContent({ html }: { html: string }) {
  return <div dangerouslySetInnerHTML={{ __html: html }} />;
}

function renderSegment(segment: BilingualSegment, value: string, suffix: string) {
  const id = segment.id ? `${segment.id}-${suffix}` : undefined;

  if (segment.kind === "heading") {
    const Heading = `h${segment.level ?? 2}` as "h2" | "h3" | "h4";

    return <Heading id={id}>{value}</Heading>;
  }

  if (segment.kind === "html") {
    return <SegmentContent html={value} />;
  }

  return <p>{value}</p>;
}

export function BilingualArticle({ segments }: BilingualArticleProps) {
  const [mode, setMode] = useState<BilingualMode>("zh");
  const articleRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;

    ensureMathJax()
      .then(async () => {
        const root = articleRef.current;
        const mathJax = (window as MathJaxWindow).MathJax;

        if (!cancelled && root && mathJax?.typesetPromise) {
          await mathJax.typesetPromise([root]);
        }
      })
      .catch(() => {
        // MathJax is progressive enhancement; leave TeX readable if the CDN fails.
      });

    return () => {
      cancelled = true;
    };
  }, [mode, segments]);

  return (
    <div className="bilingual-article" data-mode={mode} ref={articleRef}>
      <div className="bilingual-toolbar" aria-label="Language mode">
        <button
          aria-pressed={mode === "zh"}
          onClick={() => setMode("zh")}
          type="button"
        >
          <Languages size={16} strokeWidth={2} />
          中文
        </button>
        <button
          aria-pressed={mode === "both"}
          onClick={() => setMode("both")}
          type="button"
        >
          <Columns2 size={16} strokeWidth={2} />
          对照
        </button>
        <button
          aria-pressed={mode === "en"}
          onClick={() => setMode("en")}
          type="button"
        >
          EN
        </button>
      </div>

      <div className="bilingual-content">
        {segments.map((segment, index) => (
          <section className="bilingual-segment" key={segment.id ?? index}>
            {mode === "both" ? (
              <div className="bilingual-pair">
                <div>{renderSegment(segment, segment.zh, "zh")}</div>
                <div>{renderSegment(segment, segment.en, "en")}</div>
              </div>
            ) : (
              renderSegment(segment, mode === "zh" ? segment.zh : segment.en, mode)
            )}
          </section>
        ))}
      </div>
    </div>
  );
}
