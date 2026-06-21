const VIEW_COUNT_HOST = "jxtang.tech";

export function normalizeViewCountKey(value: string) {
  return value
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

export function viewCountPath(scope: string, slug: string) {
  return `${VIEW_COUNT_HOST}/${normalizeViewCountKey(scope)}/${normalizeViewCountKey(slug)}`;
}

export function viewCountTargetForEntry(entry: { slug: string; type: string }) {
  if (entry.type === "writing") {
    return {
      scope: "blog",
      slug: entry.slug
    };
  }

  if (entry.type === "reading") {
    return {
      scope: "reading",
      slug: entry.slug.replace(/^reading-/, "")
    };
  }

  return {
    scope: entry.type,
    slug: entry.slug
  };
}
