import { PaperIndex } from "@/components/paper-index";
import { profile } from "@/content/profile";

export const metadata = {
  title: "Papers"
};

export default function PapersPage() {
  const scholarLink = profile.links.find((link) => link.label === "Google Scholar");

  return (
    <div className="papers-page">
      <header className="simple-page-header">
        <p className="eyebrow">Papers</p>
        <h1>Publication index</h1>
        <p>
          Research papers, reading notes, and implementation references connected to AI
          for science, biological sequence modeling, and research tooling.
        </p>
        {scholarLink ? (
          <a className="text-link inline-link" href={scholarLink.href}>
            Google Scholar
          </a>
        ) : null}
      </header>
      <PaperIndex papers={profile.papers} />
    </div>
  );
}
