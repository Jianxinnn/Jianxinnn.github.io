import { ReadingIndex } from "@/components/reading-index";
import { profile } from "@/content/profile";

export const metadata = {
  title: "Readings"
};

export default function ReadingsPage() {
  return (
    <div className="readings-page">
      <header className="simple-page-header compact-page-header">
        <p className="eyebrow">Readings</p>
      </header>
      <ReadingIndex readings={profile.readings} />
    </div>
  );
}
