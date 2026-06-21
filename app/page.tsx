import Link from "next/link";
import { Mail } from "lucide-react";
import { BlogPostImage } from "@/components/blog-post-image";
import { EntryList } from "@/components/entry-list";
import { entries } from "@/content/entries";
import { profile } from "@/content/profile";
import { formatDate, formatEntryType, sortEntries } from "@/lib/content";

export default function HomePage() {
  const allEntries = sortEntries(entries);
  const currentWork = allEntries.slice(0, 2);
  const history = allEntries.slice(2);

  return (
    <div className="page-shell">
      <section className="home-hero">
        <div className="home-signature" aria-label={profile.name}>
          <span className="signature-letter" aria-hidden="true">J</span>
          <h1>{profile.name}</h1>
        </div>

        <section className="current-panel" aria-labelledby="current-work-heading">
          <div className="current-heading">
            <div>
              <p className="eyebrow">Current index</p>
              <h1 id="current-work-heading">Latest writing, notes, and readings</h1>
            </div>
            <Link href="/archive">Archive</Link>
          </div>

          <div className="current-work-list">
            {currentWork.map((entry, index) => (
              <article className="current-work-item" key={entry.slug}>
                {entry.image ? (
                  entry.href?.startsWith("http") ? (
                    <a
                      aria-label={entry.title}
                      className="current-work-image-link"
                      href={entry.href}
                      rel="noreferrer"
                      target="_blank"
                    >
                      <BlogPostImage
                        alt=""
                        className="current-work-image"
                        height={214}
                        priority={index === 0}
                        src={entry.image}
                        width={320}
                      />
                    </a>
                  ) : (
                    <Link
                      aria-label={entry.title}
                      className="current-work-image-link"
                      href={entry.href ?? `/archive#${entry.slug}`}
                    >
                      <BlogPostImage
                        alt=""
                        className="current-work-image"
                        height={214}
                        priority={index === 0}
                        src={entry.image}
                        width={320}
                      />
                    </Link>
                  )
                ) : null}
                <div className="current-work-copy">
                  <div className="entry-meta">
                    <time dateTime={entry.date}>{formatDate(entry.date)}</time>
                    <span aria-hidden="true">·</span>
                    <span>{formatEntryType(entry.type)}</span>
                  </div>
                  <div className="current-title-line">
                    <h2>
                      {entry.href?.startsWith("http") ? (
                        <a href={entry.href} rel="noreferrer" target="_blank">
                          {entry.title}
                        </a>
                      ) : (
                        <Link href={entry.href ?? `/archive#${entry.slug}`}>
                          {entry.title}
                        </Link>
                      )}
                    </h2>
                  </div>
                  <p>{entry.summary}</p>
                </div>
              </article>
            ))}
          </div>
        </section>
      </section>

      <section className="home-log-layout">
        <div className="recent-log">
          <div className="section-heading">
            <h2>Recent log</h2>
            <Link href="/archive">Archive</Link>
          </div>
          <EntryList entries={history} />
        </div>

        <aside className="mail-panel" aria-label="Email updates">
          <div className="mail-panel-icon" aria-hidden="true">
            <Mail size={20} strokeWidth={2} />
          </div>
          <h2>Mail</h2>
          <p>Occasional updates on research systems, notes, readings, and project logs.</p>
          <form action={`mailto:${profile.email}`} className="mail-form">
            <input aria-label="Email address" name="email" placeholder="Email address" type="email" />
            <button type="submit">Subscribe</button>
          </form>
        </aside>
      </section>
    </div>
  );
}
