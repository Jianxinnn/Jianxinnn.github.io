import { EntryList } from "@/components/entry-list";
import { entries } from "@/content/entries";
import { sortEntries } from "@/lib/content";

export const metadata = {
  title: "Notes"
};

export default function NotesPage() {
  const notes = sortEntries(entries.filter((entry) => entry.type === "note"));

  return (
    <div className="notes-page">
      <header className="simple-page-header">
        <h1>Notes</h1>
        <p>
          Short entries for formulas, small derivations, design observations, and
          research workflow fragments that do not need a full essay.
        </p>
      </header>
      <EntryList entries={notes} showImages={false} showViewCounts />
    </div>
  );
}
