"use client";

import { Menu } from "lucide-react";
import { useEffect, useState } from "react";

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
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const root = document.querySelector(targetSelector);

    if (!root) {
      return;
    }

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
  }, [targetSelector]);

  if (!headings.length) {
    return null;
  }

  return (
    <aside className={`section-nav-shell${open ? " is-open" : ""}`} aria-label="Article sections">
      <button
        aria-expanded={open}
        aria-label="Article sections"
        className="section-nav-toggle"
        onClick={() => setOpen((value) => !value)}
        type="button"
      >
        <Menu size={18} strokeWidth={2} />
      </button>
      <nav className="section-nav-panel">
        <p>Sections</p>
        {headings.map((heading) => (
          <a
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
