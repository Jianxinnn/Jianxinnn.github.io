import AdaworldLatentActions from "@/content/notes/adaworld-latent-actions/index.mdx";
import type { Note } from "@/content/notes/types";

export const notes = [
  {
    slug: "adaworld-latent-actions",
    title: "AdaWorld latent actions",
    summary:
      "A short note on action-aware pretraining as a reusable control interface for adaptable world models.",
    date: "2026-06-21",
    updated: "2026-06-21T23:59:00+08:00",
    readingTime: "2 min read",
    image: "/assets/blog/adaworld-latent-actions/fig1-framework.jpg",
    tags: ["world models", "reinforcement learning"],
    Content: AdaworldLatentActions
  }
] satisfies Note[];

export function getNote(slug: string) {
  return notes.find((note) => note.slug === slug);
}
