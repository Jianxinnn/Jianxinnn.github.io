import { profile } from "@/content/profile";

export const metadata = {
  title: "About"
};

export default function AboutPage() {
  return (
    <div className="about-page">
      <section className="about-hero">
        <img
          alt={`${profile.name} avatar`}
          className="about-avatar"
          height="176"
          src={profile.assets.avatar}
          width="176"
        />
        <div className="about-intro">
          <p className="eyebrow">About</p>
          <h1>{profile.name}</h1>
          <p>{profile.about}</p>
          <dl className="fact-list">
            {profile.facts.map((fact) => (
              <div key={fact.label}>
                <dt>{fact.label}</dt>
                <dd>{fact.value}</dd>
              </div>
            ))}
          </dl>
        </div>
      </section>

      <section className="about-block">
        <h2>Bio</h2>
        <div className="about-prose">
          {profile.bio.map((paragraph) => (
            <p key={paragraph}>{paragraph}</p>
          ))}
        </div>
      </section>

      <section className="about-block">
        <h2>Papers</h2>
        <div className="info-list">
          {profile.papers.map((paper) => (
            <article className="info-item" key={paper.title}>
              <p className="info-kicker">{paper.venue}</p>
              <h3>{paper.title}</h3>
              <p>{paper.description}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="about-block">
        <h2>Experience</h2>
        <div className="timeline-list">
          {profile.experience.map((item) => (
            <article className="timeline-item" key={`${item.role}-${item.period}`}>
              <div>
                <p className="info-kicker">{item.period}</p>
                <h3>{item.role}</h3>
                <p className="timeline-place">{item.place}</p>
              </div>
              <p>{item.description}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="about-block contact-block">
        <h2>Contact</h2>
        <div className="contact-links">
          <a href={`mailto:${profile.email}`}>{profile.email}</a>
          {profile.links.map((link) => (
            <a href={link.href} key={link.label}>
              {link.label}
            </a>
          ))}
        </div>
      </section>
    </div>
  );
}
