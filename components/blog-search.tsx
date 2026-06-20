"use client";

import { useEffect, useState } from "react";

type PagefindResult = {
  data: () => Promise<{
    excerpt: string;
    meta: {
      title?: string;
    };
    url: string;
  }>;
};

type PagefindModule = {
  search: (query: string) => Promise<{
    results: PagefindResult[];
  }>;
};

type SearchResult = {
  excerpt: string;
  title: string;
  url: string;
};

const importPagefind = new Function("path", "return import(path)") as (
  path: string
) => Promise<PagefindModule>;

export function BlogSearch() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");

  useEffect(() => {
    const trimmed = query.trim();

    if (trimmed.length < 2) {
      setResults([]);
      setStatus("idle");
      return;
    }

    let cancelled = false;
    const timeout = window.setTimeout(async () => {
      setStatus("loading");

      try {
        const pagefind = await importPagefind("/pagefind/pagefind.js");
        const search = await pagefind.search(trimmed);
        const nextResults = await Promise.all(
          search.results.slice(0, 6).map(async (result) => {
            const data = await result.data();

            return {
              excerpt: data.excerpt,
              title: data.meta.title ?? data.url,
              url: data.url
            };
          })
        );

        if (!cancelled) {
          setResults(nextResults);
          setStatus("ready");
        }
      } catch {
        if (!cancelled) {
          setResults([]);
          setStatus("error");
        }
      }
    }, 180);

    return () => {
      cancelled = true;
      window.clearTimeout(timeout);
    };
  }, [query]);

  return (
    <section className="blog-search" aria-label="Search blog">
      <input
        aria-label="Search blog"
        onChange={(event) => setQuery(event.target.value)}
        placeholder="Search blog"
        type="search"
        value={query}
      />
      {status === "ready" && results.length > 0 ? (
        <div className="blog-search-results">
          {results.map((result) => (
            <a href={result.url} key={result.url}>
              <strong>{result.title}</strong>
              <span dangerouslySetInnerHTML={{ __html: result.excerpt }} />
            </a>
          ))}
        </div>
      ) : null}
      {status === "ready" && results.length === 0 ? (
        <p className="blog-search-empty">No results.</p>
      ) : null}
    </section>
  );
}
