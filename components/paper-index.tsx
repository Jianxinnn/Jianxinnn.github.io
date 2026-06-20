import type { Paper } from "@/content/profile";

type PaperIndexProps = {
  papers: Paper[];
};

export function PaperIndex({ papers }: PaperIndexProps) {
  return (
    <div className="paper-index">
      {papers.map((paper) => (
        <article className="paper-row" key={`${paper.year}-${paper.title}`}>
          <div className="paper-year">{paper.year}</div>
          <div className="paper-main">
            <p className="paper-meta">
              {paper.type} · {paper.venue}
            </p>
            <h3>{paper.title}</h3>
            <p>{paper.description}</p>
            <div className="paper-tags" aria-label="Paper topics">
              {paper.tags.map((tag) => (
                <span key={tag}>{tag}</span>
              ))}
            </div>
            {paper.links.length > 0 ? (
              <div className="paper-links">
                {paper.links.map((link) => (
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
