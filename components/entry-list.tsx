import Link from "next/link";
import { BlogPostImage } from "@/components/blog-post-image";
import type { Entry } from "@/content/entries";
import { formatDate, formatEntryType } from "@/lib/content";

type EntryListProps = {
  entries: Entry[];
  showImages?: boolean;
};

export function EntryList({ entries, showImages = true }: EntryListProps) {
  return (
    <div className="entry-list">
      {entries.map((entry, index) => {
        const href = entry.href ?? `/archive#${entry.slug}`;

        return (
          <article className="entry-row" key={entry.slug}>
            <div className="entry-copy">
              {href.startsWith("http") ? (
                <a className="entry-title" href={href} rel="noreferrer" target="_blank">
                  {entry.title}
                </a>
              ) : (
                <Link className="entry-title" href={href} prefetch={index < 3}>
                  {entry.title}
                </Link>
              )}
              <p className="entry-summary">{entry.summary}</p>
              <div className="entry-meta">
                <time dateTime={entry.date}>{formatDate(entry.date)}</time>
                <span aria-hidden="true">·</span>
                <span>{formatEntryType(entry.type)}</span>
              </div>
            </div>
            {showImages && entry.image ? (
              <BlogPostImage
                alt=""
                className="entry-thumb"
                height={214}
                src={entry.image}
                width={320}
              />
            ) : null}
          </article>
        );
      })}
    </div>
  );
}
