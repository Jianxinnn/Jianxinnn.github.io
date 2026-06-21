"use client";

import { useEffect, useRef } from "react";
import { ensureMathJax, type MathJaxWindow } from "@/components/mathjax";

type MathBlockProps = {
  className?: string;
  tex: string;
};

export function MathBlock({ className, tex }: MathBlockProps) {
  const blockRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;

    ensureMathJax()
      .then(async () => {
        const root = blockRef.current;
        const mathJax = (window as MathJaxWindow).MathJax;

        if (!cancelled && root && mathJax?.typesetPromise) {
          await mathJax.typesetPromise([root]);
        }
      })
      .catch(() => {
        // Leave readable TeX in place if MathJax cannot load.
      });

    return () => {
      cancelled = true;
    };
  }, [tex]);

  return (
    <div className={["math-block", className].filter(Boolean).join(" ")} ref={blockRef}>
      {`\\[${tex}\\]`}
    </div>
  );
}
