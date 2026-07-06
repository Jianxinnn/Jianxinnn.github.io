"use client";

import { useEffect, useRef } from "react";

type ReadingProgressProps = {
  targetSelector?: string;
};

function getProgress(targetSelector: string) {
  const target = document.querySelector<HTMLElement>(targetSelector);
  const documentHeight = document.documentElement.scrollHeight - window.innerHeight;

  if (!target) {
    return documentHeight > 0 ? window.scrollY / documentHeight : 0;
  }

  const targetTop = target.getBoundingClientRect().top + window.scrollY;
  const targetHeight = target.scrollHeight;
  const start = Math.max(0, targetTop - window.innerHeight * 0.18);
  const end = Math.max(start + 1, targetTop + targetHeight - window.innerHeight * 0.68);

  return (window.scrollY - start) / (end - start);
}

export function ReadingProgress({ targetSelector = ".mdx-body" }: ReadingProgressProps) {
  const barRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let frame = 0;

    const sync = () => {
      window.cancelAnimationFrame(frame);
      frame = window.requestAnimationFrame(() => {
        const rawProgress = getProgress(targetSelector);
        const progress = Math.max(0, Math.min(1, rawProgress));

        if (barRef.current) {
          barRef.current.style.transform = `scaleX(${progress})`;
        }
      });
    };

    sync();
    window.addEventListener("scroll", sync, { passive: true });
    window.addEventListener("resize", sync);

    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener("scroll", sync);
      window.removeEventListener("resize", sync);
    };
  }, [targetSelector]);

  return <div aria-hidden="true" className="reading-progress-bar" ref={barRef} />;
}
