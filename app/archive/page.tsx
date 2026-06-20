import { entries } from "@/content/entries";
import { formatDate, groupEntriesByMonth } from "@/lib/content";

export const metadata = {
  title: "Archive"
};

export default function ArchivePage() {
  const groups = groupEntriesByMonth(entries);

  return (
    <div className="archive-page">
      {groups.map((group, index) => (
        <section className="archive-group" key={group.label}>
          {index > 0 ? <h2>{group.label}</h2> : null}
          {group.entries.map((entry) => (
            <article className="archive-entry" id={entry.slug} key={entry.slug}>
              <h3>{entry.title}</h3>
              <p>{entry.summary}</p>
              <div className="entry-meta">
                <time dateTime={entry.date}>{formatDate(entry.date)}</time>
                <span aria-hidden="true">·</span>
                <span>{entry.type}</span>
              </div>
            </article>
          ))}
        </section>
      ))}
    </div>
  );
}
