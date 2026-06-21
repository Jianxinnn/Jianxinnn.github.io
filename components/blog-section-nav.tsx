"use client";

import type { MouseEvent } from "react";
import { useEffect, useRef, useState } from "react";

type Heading = {
  depth: number;
  id: string;
  text: string;
};

type BlogSectionNavProps = {
  targetSelector?: string;
};

const IMAGE_WARMUP_TIMEOUT = 700;

function slugifyHeading(value: string) {
  return value
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9\u3400-\u9fff]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function getScrollOffset() {
  const headerHeight =
    document.querySelector<HTMLElement>(".site-header")?.getBoundingClientRect().height ?? 64;

  return headerHeight + 34;
}

function waitForImage(image: HTMLImageElement) {
  if (image.complete) {
    return image.decode ? image.decode().catch(() => undefined) : Promise.resolve();
  }

  return new Promise<void>((resolve) => {
    const cleanup = () => {
      image.removeEventListener("load", cleanup);
      image.removeEventListener("error", cleanup);
      resolve();
    };

    image.addEventListener("load", cleanup, { once: true });
    image.addEventListener("error", cleanup, { once: true });
  });
}

async function warmImagesBeforeTarget(root: Element | null, target: HTMLElement) {
  if (!root) {
    return;
  }

  const images = Array.from(root.querySelectorAll<HTMLImageElement>("img[loading='lazy']")).filter(
    (image) => image.compareDocumentPosition(target) & Node.DOCUMENT_POSITION_FOLLOWING
  );

  if (!images.length) {
    return;
  }

  images.forEach((image) => {
    image.loading = "eager";
    image.decoding = "async";
  });

  await Promise.race([
    Promise.all(images.map((image) => waitForImage(image))),
    new Promise((resolve) => window.setTimeout(resolve, IMAGE_WARMUP_TIMEOUT))
  ]);
}

function scrollWindowTo(top: number, behavior: ScrollBehavior) {
  if (behavior === "smooth") {
    window.scrollTo({ behavior, top });
    return;
  }

  const html = document.documentElement;
  const previousScrollBehavior = html.style.scrollBehavior;

  html.style.scrollBehavior = "auto";
  window.scrollTo({ behavior: "auto", top });
  window.requestAnimationFrame(() => {
    html.style.scrollBehavior = previousScrollBehavior;
  });
}

export function BlogSectionNav({ targetSelector = ".mdx-body" }: BlogSectionNavProps) {
  const [headings, setHeadings] = useState<Heading[]>([]);
  const [activeId, setActiveId] = useState<string>();
  const [open, setOpen] = useState(false);
  const panelRef = useRef<HTMLElement>(null);
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

    let frame = 0;

    const syncActiveHeading = () => {
      const anchorLine = window.innerHeight * 0.28;
      let nextActiveId = elements[0]?.id;

      for (const element of elements) {
        if (element.getBoundingClientRect().top <= anchorLine) {
          nextActiveId = element.id;
        } else {
          break;
        }
      }

      if (nextActiveId) {
        setActiveId(nextActiveId);
      }
    };

    const scheduleSync = () => {
      window.cancelAnimationFrame(frame);
      frame = window.requestAnimationFrame(syncActiveHeading);
    };

    syncActiveHeading();
    window.addEventListener("scroll", scheduleSync, { passive: true });
    window.addEventListener("resize", scheduleSync);

    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener("scroll", scheduleSync);
      window.removeEventListener("resize", scheduleSync);
    };
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

  useEffect(() => {
    const panel = panelRef.current;
    const activeLink = panel?.querySelector<HTMLElement>('a[aria-current="location"]');

    if (!panel || !activeLink) {
      return;
    }

    panel.scrollTop =
      activeLink.offsetTop - panel.clientHeight / 2 + activeLink.clientHeight / 2;
  }, [activeId, headings.length]);

  const handleHeadingClick = async (event: MouseEvent<HTMLAnchorElement>, id: string) => {
    if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey || event.button !== 0) {
      return;
    }

    const target = document.getElementById(id);

    if (!target) {
      return;
    }

    event.preventDefault();
    setActiveId(id);
    setOpen(true);

    await warmImagesBeforeTarget(document.querySelector(targetSelector), target);

    const getTargetTop = () =>
      Math.max(0, target.getBoundingClientRect().top + window.scrollY - getScrollOffset());
    const top = getTargetTop();
    const distance = Math.abs(window.scrollY - top);
    const encodedHash = `#${encodeURIComponent(id)}`;

    window.history.pushState(null, "", encodedHash);
    scrollWindowTo(top, distance > window.innerHeight * 3 ? "auto" : "smooth");
  };

  if (!headings.length) {
    return null;
  }

  return (
    <aside
      className={`section-nav-shell${open ? " is-open" : ""}`}
      aria-label="Article sections"
      onFocus={() => setOpen(true)}
      onMouseEnter={() => setOpen(true)}
      ref={shellRef}
    >
      <nav className="section-nav-panel" id="article-section-nav" ref={panelRef}>
        <div className="section-nav-heading">
          <p>目录</p>
          <span>{headings.length}</span>
        </div>
        {headings.map((heading) => (
          <a
            aria-label={heading.text}
            aria-current={activeId === heading.id ? "location" : undefined}
            className={`section-nav-depth-${Math.min(heading.depth, 3)}`}
            href={`#${heading.id}`}
            key={heading.id}
            onClick={(event) => handleHeadingClick(event, heading.id)}
            title={heading.text}
          >
            <span aria-hidden="true" className="section-nav-marker" />
            <span className="section-nav-label">{heading.text}</span>
          </a>
        ))}
      </nav>
    </aside>
  );
}
