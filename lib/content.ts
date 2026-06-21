import type { Entry } from "@/content/entries";

function entrySortTime(entry: Entry) {
  return new Date(entry.updated ?? entry.date).getTime();
}

export function sortEntries(items: Entry[]) {
  return [...items].sort((a, b) => {
    const timeDifference = entrySortTime(b) - entrySortTime(a);

    if (timeDifference !== 0) {
      return timeDifference;
    }

    return a.slug.localeCompare(b.slug);
  });
}

export function formatDate(value: string) {
  return new Intl.DateTimeFormat("en", {
    month: "short",
    day: "numeric",
    year: "numeric"
  }).format(new Date(value));
}

export function formatEntryType(type: Entry["type"]) {
  const labels: Record<Entry["type"], string> = {
    project: "Project",
    note: "Note",
    writing: "Blog",
    reading: "Reading",
    talk: "Talk",
    publication: "Publication"
  };

  return labels[type];
}

export function monthLabel(value: string) {
  return new Intl.DateTimeFormat("en", {
    month: "long",
    year: "numeric"
  })
    .format(new Date(value))
    .toUpperCase();
}

export function groupEntriesByMonth(items: Entry[]) {
  return sortEntries(items).reduce<Array<{ label: string; entries: Entry[] }>>(
    (groups, entry) => {
      const label = monthLabel(entry.date);
      const group = groups.find((item) => item.label === label);

      if (group) {
        group.entries.push(entry);
      } else {
        groups.push({ label, entries: [entry] });
      }

      return groups;
    },
    []
  );
}
