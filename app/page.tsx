import Image from "next/image";
import Link from "next/link";
import { EntryList } from "@/components/entry-list";
import { entries } from "@/content/entries";
import { profile } from "@/content/profile";
import { formatDate, sortEntries } from "@/lib/content";

export default function HomePage() {
  const sorted = sortEntries(entries);
  const pinned = sorted.filter((entry) => entry.pinned).slice(0, 3);
  const recent = sorted.filter((entry) => !pinned.some((item) => item.slug === entry.slug));
  const scholarLink = profile.links.find((link) => link.label === "Google Scholar");

  return (
    <div className="page-shell">
      <section className="home-hero">
        <div className="home-copy">
          <p className="eyebrow">{profile.shortRole}</p>
          <h1>{profile.name}</h1>
          <p>{profile.intro}</p>
          <div className="home-actions">
            <Link className="primary-button" href="/about">
              About
            </Link>
            {scholarLink ? (
              <a className="secondary-button" href={scholarLink.href}>
                Google Scholar
              </a>
            ) : null}
          </div>
        </div>
        <aside className="identity-panel" aria-label="Profile summary">
          <img
            alt={`${profile.name} avatar`}
            className="identity-photo"
            height="144"
            src={profile.assets.avatar}
            width="112"
          />
          <dl className="identity-list">
            {profile.facts.map((fact) => (
              <div key={fact.label}>
                <dt>{fact.label}</dt>
                <dd>{fact.value}</dd>
              </div>
            ))}
          </dl>
        </aside>
      </section>

      <section className="work-ledger">
        <div className="ledger-heading">
          <div>
            <p className="eyebrow">Current work</p>
            <h2>Research systems and working notes</h2>
          </div>
          <Link href="/archive">Archive</Link>
        </div>

        <div className="ledger-grid">
          {pinned.map((entry) => (
            <article className="work-card" key={entry.slug}>
              {entry.image ? (
                <Image
                  alt=""
                  className="work-card-image"
                  height={214}
                  src={entry.image}
                  width={320}
                />
              ) : null}
              <div>
                <div className="entry-meta">
                  <time dateTime={entry.date}>{formatDate(entry.date)}</time>
                  <span aria-hidden="true">·</span>
                  <span>{entry.collaborators}</span>
                </div>
                <h3>{entry.title}</h3>
                <p>{entry.summary}</p>
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="recent-log">
        <div className="section-heading">
          <h2>Recent log</h2>
          <Link href="/notes">Notes</Link>
        </div>
        <EntryList entries={recent} />
      </section>
    </div>
  );
}
