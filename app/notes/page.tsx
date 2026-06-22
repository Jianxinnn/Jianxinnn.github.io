import { EntryList } from "@/components/entry-list";
import { notes } from "@/content/notes/notes";
import { sortEntries } from "@/lib/content";

export const metadata = {
  title: "Notes"
};

export default function NotesPage() {
  const noteEntries = sortEntries(
    notes.map((note) => ({
      slug: note.slug,
      title: note.title,
      summary: note.summary,
      date: note.date,
      type: "note" as const,
      collaborators: note.tags?.join(" / ") ?? note.readingTime,
      image: note.image,
      href: `/notes/${note.slug}/`,
      updated: note.updated
    }))
  );

  return (
    <div className="notes-page">
      <header className="simple-page-header">
        <h1>Notes</h1>
        <p>
          Short entries for formulas, small derivations, design observations, and
          research workflow fragments that do not need a full essay.
        </p>
      </header>
      <EntryList entries={noteEntries} showImages={false} showViewCounts />
    </div>
  );
}
