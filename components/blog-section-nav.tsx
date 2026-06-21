"use client";

import { Menu, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";

type Heading = {
  depth: number;
  id: string;
  text: string;
};

type BlogSectionNavProps = {
  targetSelector?: string;
};

function slugifyHeading(value: string) {
  return value
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9\u3400-\u9fff]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

export function BlogSectionNav({ targetSelector = ".mdx-body" }: BlogSectionNavProps) {
  const [headings, setHeadings] = useState<Heading[]>([]);
  const [activeId, setActiveId] = useState<string>();
  const [open, setOpen] = useState(false);
  const shellRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const root = document.querySelector(targetSelector);

    if (!root) {
      return;
    }

    const rebuild = () => {
      const usedIds = new Set<string>();
      const nextHeadings = Array.from(root.querySelectorAll("h1, h2, h3"))
        .map((element) => {
          const text = element.textContent?.trim() ?? "";

          if (!text) {
            return undefined;
          }

          let id = element.id || slugifyHeading(text) || "section";
          let suffix = 2;

          while (usedIds.has(id)) {
            id = `${id}-${suffix}`;
            suffix += 1;
          }

          usedIds.add(id);
          element.id = id;

          return {
            depth: Number(element.tagName.slice(1)),
            id,
            text
          };
        })
        .filter(Boolean) as Heading[];

      setHeadings(nextHeadings);
      setActiveId((current) =>
        nextHeadings.some((heading) => heading.id === current) ? current : nextHeadings[0]?.id
      );
    };

    let frame = 0;
    const scheduleRebuild = () => {
      window.cancelAnimationFrame(frame);
      frame = window.requestAnimationFrame(rebuild);
    };

    rebuild();

    const observer = new MutationObserver(scheduleRebuild);
    observer.observe(root, { childList: true, subtree: true });

    return () => {
      window.cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, [targetSelector]);

  useEffect(() => {
    if (!headings.length) {
      return;
    }

    const elements = headings
      .map((heading) => document.getElementById(heading.id))
      .filter(Boolean) as HTMLElement[];

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];

        if (visible?.target.id) {
          setActiveId(visible.target.id);
        }
      },
      { rootMargin: "-20% 0px -68% 0px", threshold: [0, 1] }
    );

    elements.forEach((element) => observer.observe(element));

    return () => observer.disconnect();
  }, [headings]);

  useEffect(() => {
    if (!open) {
      return;
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpen(false);
      }
    };

    const onPointerDown = (event: PointerEvent) => {
      if (shellRef.current && !shellRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    };

    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("pointerdown", onPointerDown);

    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("pointerdown", onPointerDown);
    };
  }, [open]);

  if (!headings.length) {
    return null;
  }

  return (
    <aside
      className={`section-nav-shell${open ? " is-open" : ""}`}
      aria-label="Article sections"
      ref={shellRef}
    >
      <button
        aria-controls="article-section-nav"
        aria-expanded={open}
        aria-label={open ? "Close article sections" : "Open article sections"}
        className="section-nav-toggle"
        onClick={() => setOpen((value) => !value)}
        type="button"
      >
        {open ? <X size={17} strokeWidth={2} /> : <Menu size={17} strokeWidth={2} />}
        <span>目录</span>
      </button>
      <nav className="section-nav-panel" id="article-section-nav">
        <div className="section-nav-heading">
          <p>Contents</p>
          <span>{headings.length}</span>
        </div>
        {headings.map((heading) => (
          <a
            aria-current={activeId === heading.id ? "location" : undefined}
            className={`section-nav-depth-${Math.min(heading.depth, 3)}`}
            href={`#${heading.id}`}
            key={heading.id}
            onClick={() => setOpen(false)}
          >
            {heading.text}
          </a>
        ))}
      </nav>
    </aside>
  );
}
