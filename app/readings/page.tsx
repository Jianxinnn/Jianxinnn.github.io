import { ReadingIndex } from "@/components/reading-index";
import { profile } from "@/content/profile";

export const metadata = {
  title: "Readings"
};

export default function ReadingsPage() {
  const scholarLink = profile.links.find((link) => link.label === "Google Scholar");

  return (
    <div className="readings-page">
      <header className="simple-page-header">
        <p className="eyebrow">Readings</p>
        <h1>Paper reading notes</h1>
        <p>
          A separate index for paper reading records: core claims, method sketches,
          useful equations, implementation notes, and follow-up questions.
        </p>
        {scholarLink ? (
          <a className="text-link inline-link" href={scholarLink.href}>
            Google Scholar
          </a>
        ) : null}
      </header>
      <ReadingIndex readings={profile.readings} />
    </div>
  );
}
