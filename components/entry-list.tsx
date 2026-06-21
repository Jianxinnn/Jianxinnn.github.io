import Image from "next/image";
import Link from "next/link";
import { ViewCountBadge } from "@/components/view-count-badge";
import type { Entry } from "@/content/entries";
import { formatDate, formatEntryType } from "@/lib/content";

type EntryListProps = {
  entries: Entry[];
  showImages?: boolean;
  showViewCounts?: boolean;
};

export function EntryList({
  entries,
  showImages = true,
  showViewCounts = false
}: EntryListProps) {
  return (
    <div className="entry-list">
      {entries.map((entry) => (
        <article className="entry-row" key={entry.slug}>
          <div className="entry-copy">
            <Link className="entry-title" href={entry.href ?? `/archive#${entry.slug}`}>
              {entry.title}
            </Link>
            <p className="entry-summary">{entry.summary}</p>
            <div className="entry-meta">
              <time dateTime={entry.date}>{formatDate(entry.date)}</time>
              <span aria-hidden="true">·</span>
              <span>{formatEntryType(entry.type)}</span>
              {showViewCounts ? (
                <>
                  <span aria-hidden="true">·</span>
                  <ViewCountBadge scope={entry.type} slug={entry.slug} />
                </>
              ) : null}
            </div>
          </div>
          {showImages && entry.image ? (
            <Image
              alt=""
              className="entry-thumb"
              height={214}
              src={entry.image}
              width={320}
            />
          ) : null}
        </article>
      ))}
    </div>
  );
}
