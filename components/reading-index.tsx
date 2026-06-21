import type { Reading } from "@/content/profile";

type ReadingIndexProps = {
  readings: Reading[];
};

export function ReadingIndex({ readings }: ReadingIndexProps) {
  return (
    <div className="reading-index">
      {readings.map((reading) => (
        <article className="reading-row" key={`${reading.year}-${reading.title}`}>
          <div className="reading-year">{reading.year}</div>
          <div className="reading-main">
            <p className="reading-meta">
              {reading.type} · {reading.venue}
            </p>
            <h3>{reading.title}</h3>
            <p>{reading.description}</p>
            <div className="reading-tags" aria-label="Reading topics">
              {reading.tags.map((tag) => (
                <span key={tag}>{tag}</span>
              ))}
            </div>
            {reading.links.length > 0 ? (
              <div className="reading-links">
                {reading.links.map((link) => (
                  <a href={link.href} key={link.label}>
                    {link.label}
                  </a>
                ))}
              </div>
            ) : null}
          </div>
        </article>
      ))}
    </div>
  );
}
