import type { ReactNode } from "react";

export type BilingualSegment = {
  en: string;
  id?: string;
  kind?: "heading" | "paragraph" | "html";
  level?: 2 | 3 | 4;
  zh: string;
};

export type BilingualMode = "zh" | "both" | "en";

type CalloutType = "key-idea" | "remark" | "summary";

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

    return <Heading id={id} key={suffix}>{value}</Heading>;
  }

  if (segment.kind === "html") {
    return <SegmentContent html={value} key={suffix} />;
  }

  return <p key={suffix}>{value}</p>;
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

export function BilingualArticleContent({
  mode,
  segments
}: {
  mode: BilingualMode;
  segments: BilingualSegment[];
}) {
  return (
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
  );
}
