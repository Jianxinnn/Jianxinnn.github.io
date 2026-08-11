"use client";

import { useEffect, useRef } from "react";
import ReactMarkdown, { defaultUrlTransform } from "react-markdown";
import rehypeRaw from "rehype-raw";
import remarkGfm from "remark-gfm";
import { ensureMathJax, type MathJaxWindow } from "@/components/mathjax";

type ProtectedArticleBodyProps = {
  markdown: string;
};

function protectedUrlTransform(url: string, key: string) {
  if (key === "src" && /^data:image\/(?:png|jpe?g|webp|gif);base64,/i.test(url)) {
    return url;
  }

  return defaultUrlTransform(url);
}

export function ProtectedArticleBody({ markdown }: ProtectedArticleBodyProps) {
  const articleBodyRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;

    ensureMathJax()
      .then(async () => {
        const root = articleBodyRef.current;
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
  }, [markdown]);

  return (
    <div className="mdx-body protected-article-body" ref={articleBodyRef}>
      <ReactMarkdown
        components={{
          img: ({ node: _node, ...props }) => (
            <img {...props} decoding="async" loading="lazy" />
          )
        }}
        rehypePlugins={[rehypeRaw]}
        remarkPlugins={[remarkGfm]}
        urlTransform={protectedUrlTransform}
      >
        {markdown}
      </ReactMarkdown>
    </div>
  );
}
