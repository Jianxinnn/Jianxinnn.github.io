import Image from "next/image";
import Link from "next/link";
import { EntryList } from "@/components/entry-list";
import { ProfileSidebar } from "@/components/profile-sidebar";
import { entries } from "@/content/entries";
import { profile } from "@/content/profile";
import { formatDate, sortEntries } from "@/lib/content";

export default function HomePage() {
  const sorted = sortEntries(entries);
  const featured = sorted.find((entry) => entry.featured) ?? sorted[0];
  const remaining = sorted.filter((entry) => entry.slug !== featured.slug);

  return (
    <div className="page-shell">
      <section className="featured">
        <div className="featured-image-wrap">
          <Image
            alt=""
            className="featured-image"
            height={640}
            priority
            src={featured.image ?? "/assets/visuals/profile-field.png"}
            width={960}
          />
        </div>
        <div className="featured-copy">
          <p className="eyebrow">{profile.role}</p>
          <h1>{featured.title}</h1>
          <p className="featured-summary">{featured.summary}</p>
          <div className="entry-meta centered">
            <time dateTime={featured.date}>{formatDate(featured.date)}</time>
            <span aria-hidden="true">·</span>
            <span>{featured.collaborators}</span>
          </div>
          <Link className="text-link" href="/about">
            Read profile context
          </Link>
        </div>
      </section>

      <section className="content-grid">
        <div>
          <div className="section-heading">
            <h2>Selected work</h2>
            <Link href="/archive">View archive</Link>
          </div>
          <EntryList entries={remaining} />
        </div>
        <ProfileSidebar />
      </section>
    </div>
  );
}
