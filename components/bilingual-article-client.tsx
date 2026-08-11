"use client";

import { Columns2, Languages } from "lucide-react";
import type { ReactNode } from "react";
import { useEffect, useRef, useState } from "react";
import {
  BilingualArticleContent,
  type BilingualMode,
  type BilingualSegment
} from "@/components/bilingual-article-content";
import { ensureMathJax, type MathJaxWindow } from "@/components/mathjax";

type BilingualArticleClientProps = {
  children: ReactNode;
  className?: string;
  segmentsPath: string;
};

const MIN_ARTICLE_IMAGE_SIZE = 80;
const segmentRequests = new Map<string, Promise<BilingualSegment[]>>();

function loadSegments(path: string) {
  const cached = segmentRequests.get(path);

  if (cached) {
    return cached;
  }

  const request = fetch(path)
    .then((response) => {
      if (!response.ok) {
        throw new Error("Bilingual article data was not found.");
      }

      return response.json();
    })
    .then((value: unknown) => {
      if (!Array.isArray(value)) {
        throw new Error("Bilingual article data is invalid.");
      }

      return value as BilingualSegment[];
    });

  segmentRequests.set(path, request);
  return request;
}

export function BilingualArticleClient({
  children,
  className,
  segmentsPath
}: BilingualArticleClientProps) {
  const [mode, setMode] = useState<BilingualMode>("zh");
  const [segments, setSegments] = useState<BilingualSegment[]>();
  const [loadingMode, setLoadingMode] = useState<Exclude<BilingualMode, "zh">>();
  const [loadError, setLoadError] = useState("");
  const articleRef = useRef<HTMLDivElement>(null);

  const warmSegments = () => {
    void loadSegments(segmentsPath).catch(() => undefined);
  };

  const selectMode = async (nextMode: BilingualMode) => {
    setLoadError("");

    if (nextMode === "zh" || segments) {
      setMode(nextMode);
      return;
    }

    setLoadingMode(nextMode);

    try {
      const loaded = await loadSegments(segmentsPath);
      setSegments(loaded);
      setMode(nextMode);
    } catch {
      segmentRequests.delete(segmentsPath);
      setLoadError("Language view could not be loaded. Please try again.");
    } finally {
      setLoadingMode(undefined);
    }
  };

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

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      window.dispatchEvent(new Event("article-content-change"));
    });

    return () => window.cancelAnimationFrame(frame);
  }, [mode, segments]);

  return (
    <div
      aria-busy={loadingMode ? true : undefined}
      className={["bilingual-article", className].filter(Boolean).join(" ")}
      data-mode={mode}
      ref={articleRef}
    >
      <div className="bilingual-toolbar" aria-label="Language mode">
        <button aria-pressed={mode === "zh"} onClick={() => void selectMode("zh")} type="button">
          <Languages size={16} strokeWidth={2} />
          中文
        </button>
        <button
          aria-pressed={mode === "both"}
          onClick={() => void selectMode("both")}
          onFocus={warmSegments}
          onMouseEnter={warmSegments}
          type="button"
        >
          <Columns2 size={16} strokeWidth={2} />
          {loadingMode === "both" ? "加载…" : "对照"}
        </button>
        <button
          aria-pressed={mode === "en"}
          onClick={() => void selectMode("en")}
          onFocus={warmSegments}
          onMouseEnter={warmSegments}
          type="button"
        >
          {loadingMode === "en" ? "Loading…" : "EN"}
        </button>
      </div>

      {loadError ? <p className="bilingual-load-error" role="status">{loadError}</p> : null}
      {mode === "zh" || !segments ? (
        children
      ) : (
        <BilingualArticleContent mode={mode} segments={segments} />
      )}
    </div>
  );
}
