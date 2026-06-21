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

