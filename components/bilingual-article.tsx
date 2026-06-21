"use client";

import { Columns2, Languages } from "lucide-react";
import type { ReactNode } from "react";
import { useEffect, useRef, useState } from "react";
import { ensureMathJax, type MathJaxWindow } from "@/components/mathjax";

export type BilingualSegment = {
  en: string;
  id?: string;
  kind?: "heading" | "paragraph" | "html";
  level?: 2 | 3 | 4;
  zh: string;
};

type BilingualArticleProps = {
  className?: string;
  segments: BilingualSegment[];
};

type BilingualMode = "zh" | "both" | "en";
type CalloutType = "key-idea" | "remark" | "summary";

const MIN_ARTICLE_IMAGE_SIZE = 80;

function SegmentContent({ html }: { html: string }) {
  return <div dangerouslySetInnerHTML={{ __html: html }} />;
}

function stripHtml(value: string) {
  return value
    .replace(/<script[\s\S]*?<\/script>/gi, "")
    .replace(/<style[\s\S]*?<\/style>/gi, "")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function isImageOnlySegment(segment: BilingualSegment) {
  const values = [segment.zh, segment.en];

  return values.every((value) => /<img\b/i.test(value) && stripHtml(value) === "");
}

function hasDisplayMath(segment: BilingualSegment) {
  return /\$\$|\\\[/.test(segment.zh) || /\$\$|\\\[/.test(segment.en);
}

function getCalloutType(segment: BilingualSegment): CalloutType | undefined {
  const zhText = stripHtml(segment.zh);
  const enText = stripHtml(segment.en);

  if (
    /^(关键思想|关键想法|关键理念)\s*\d+/i.test(zhText) ||
    /^Key Idea\s*\d+/i.test(enText)
  ) {
    return "key-idea";
  }

  if (/^备注\s*\d+/i.test(zhText) || /^Remark\s*\d+/i.test(enText)) {
    return "remark";
  }

  if (/^(总结|摘要)\s*\d+/i.test(zhText) || /^Summary\s*\d+/i.test(enText)) {
    return "summary";
  }

  return undefined;
}

function isFigureCaption(segment: BilingualSegment) {
  const text = `${stripHtml(segment.zh)} ${stripHtml(segment.en)}`;

  return /(^|\s)(图|Figure)\s*\d+[:：]/i.test(text);
}

function canUseAsCalloutBody(segment?: BilingualSegment) {
  return Boolean(
    segment &&
      !isImageOnlySegment(segment) &&
      !getCalloutType(segment) &&
      segment.kind !== "heading" &&
      !/<h[1-6]\b/i.test(segment.zh) &&
      !/<h[1-6]\b/i.test(segment.en)
  );
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

function renderSegmentStack(segments: BilingualSegment[], mode: BilingualMode, keyPrefix: string) {
  if (mode === "both") {
    return (
      <div className="bilingual-pair">
        <div>
          {segments.map((segment, index) =>
            renderSegment(segment, segment.zh, `${keyPrefix}-${index}-zh`)
          )}
        </div>
        <div>
          {segments.map((segment, index) =>
            renderSegment(segment, segment.en, `${keyPrefix}-${index}-en`)
          )}
        </div>
      </div>
    );
  }

  return segments.map((segment, index) =>
    renderSegment(segment, mode === "zh" ? segment.zh : segment.en, `${keyPrefix}-${index}-${mode}`)
  );
}

function renderCalloutStack(segments: BilingualSegment[], mode: BilingualMode, keyPrefix: string) {
  const [titleSegment, ...bodySegments] = segments;

  if (mode === "both") {
    return (
      <div className="bilingual-pair">
        <div>
          <div className="lecture-callout-title">
            {renderSegment(titleSegment, titleSegment.zh, `${keyPrefix}-title-zh`)}
          </div>
          {bodySegments.length ? (
            <div className="lecture-callout-body">
              {bodySegments.map((segment, index) =>
                renderSegment(segment, segment.zh, `${keyPrefix}-body-${index}-zh`)
              )}
            </div>
          ) : null}
        </div>
        <div>
          <div className="lecture-callout-title">
            {renderSegment(titleSegment, titleSegment.en, `${keyPrefix}-title-en`)}
          </div>
          {bodySegments.length ? (
            <div className="lecture-callout-body">
              {bodySegments.map((segment, index) =>
                renderSegment(segment, segment.en, `${keyPrefix}-body-${index}-en`)
              )}
            </div>
          ) : null}
        </div>
      </div>
    );
  }

  return (
    <>
      <div className="lecture-callout-title">
        {renderSegment(
          titleSegment,
          mode === "zh" ? titleSegment.zh : titleSegment.en,
          `${keyPrefix}-title-${mode}`
        )}
      </div>
      {bodySegments.length ? (
        <div className="lecture-callout-body">
          {bodySegments.map((segment, index) =>
            renderSegment(
              segment,
              mode === "zh" ? segment.zh : segment.en,
              `${keyPrefix}-body-${index}-${mode}`
            )
          )}
        </div>
      ) : null}
    </>
  );
}

function renderImageGroup(segments: BilingualSegment[], mode: BilingualMode) {
  const sourceMode = mode === "en" ? "en" : "zh";

  return (
    <div className={`lecture-figure-grid is-count-${Math.min(segments.length, 6)}`}>
      {segments.map((segment, index) => (
        <div className="lecture-figure-cell" key={segment.id ?? index}>
          <SegmentContent html={sourceMode === "zh" ? segment.zh : segment.en} />
        </div>
      ))}
    </div>
  );
}

export function BilingualArticle({ className, segments }: BilingualArticleProps) {
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

  useEffect(() => {
    const root = articleRef.current;

    if (!root) {
      return;
    }

    const hideTinyImage = (image: HTMLImageElement) => {
      if (
        image.complete &&
        image.naturalWidth > 0 &&
        image.naturalWidth < MIN_ARTICLE_IMAGE_SIZE &&
        image.naturalHeight < MIN_ARTICLE_IMAGE_SIZE
      ) {
        image.closest(".lecture-figure-strip")?.classList.add("is-artifact");
      }
    };

    const images = Array.from(root.querySelectorAll<HTMLImageElement>(".lecture-figure-strip img"));
    const cleanups = images.map((image) => {
      hideTinyImage(image);
      const onLoad = () => hideTinyImage(image);

      image.addEventListener("load", onLoad);

      return () => image.removeEventListener("load", onLoad);
    });

    return () => cleanups.forEach((cleanup) => cleanup());
  }, [mode, segments]);

  return (
    <div
      className={["bilingual-article", className].filter(Boolean).join(" ")}
      data-mode={mode}
      ref={articleRef}
    >
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
        {segments.reduce<ReactNode[]>((nodes, segment, index) => {
          if (index > 0 && isImageOnlySegment(segments[index - 1])) {
            return nodes;
          }

          if (isImageOnlySegment(segment)) {
            const imageGroup = [segment];
            let cursor = index + 1;

            while (segments[cursor] && isImageOnlySegment(segments[cursor])) {
              imageGroup.push(segments[cursor]);
              cursor += 1;
            }

            nodes.push(
              <section
                className={`bilingual-segment lecture-figure-strip${
                  imageGroup.length === 1 ? " is-single" : ""
                }`}
                key={segment.id ?? index}
              >
                {renderImageGroup(imageGroup, mode)}
              </section>
            );

            return nodes;
          }

          const calloutType = getCalloutType(segment);

          if (calloutType) {
            const calloutSegments = canUseAsCalloutBody(segments[index + 1])
              ? [segment, segments[index + 1]]
              : [segment];

            nodes.push(
              <section
                className={`bilingual-segment lecture-callout lecture-callout-${calloutType}`}
                key={segment.id ?? index}
              >
                {renderCalloutStack(calloutSegments, mode, `callout-${index}`)}
              </section>
            );

            return nodes;
          }

          if (index > 0 && getCalloutType(segments[index - 1]) && canUseAsCalloutBody(segment)) {
            return nodes;
          }

          nodes.push(
            <section
              className={[
                "bilingual-segment",
                isFigureCaption(segment) ? "lecture-caption" : "",
                hasDisplayMath(segment) ? "lecture-display-math" : ""
              ]
                .filter(Boolean)
                .join(" ")}
              key={segment.id ?? index}
            >
              {renderSegmentStack([segment], mode, `segment-${index}`)}
            </section>
          );

          return nodes;
        }, [])}
      </div>
    </div>
  );
}
