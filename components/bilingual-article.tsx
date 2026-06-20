"use client";

import { Columns2, Languages } from "lucide-react";
import { useState } from "react";

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

  return (
    <div className="bilingual-article" data-mode={mode}>
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
