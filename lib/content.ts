import type { Entry } from "@/content/entries";

export function sortEntries(items: Entry[]) {
  return [...items].sort(
    (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()
  );
}

export function formatDate(value: string) {
  return new Intl.DateTimeFormat("en", {
    month: "short",
    day: "numeric",
    year: "numeric"
  }).format(new Date(value));
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
