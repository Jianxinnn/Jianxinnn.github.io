import { blogPosts } from "@/content/blog/posts";

export type EntryType = "project" | "note" | "writing" | "talk" | "publication";

export type Entry = {
  slug: string;
  title: string;
  summary: string;
  date: string;
  type: EntryType;
  collaborators?: string;
  image?: string;
  href?: string;
  pinned?: boolean;
};

const blogEntries: Entry[] = blogPosts.map((post) => ({
  slug: post.slug,
  title: post.title,
  summary: post.summary,
  date: post.date,
  type: "writing",
  collaborators: post.tags?.join(" / ") ?? post.readingTime,
  image: post.image,
  href: post.href,
  pinned: post.featured
}));

const standaloneEntries: Entry[] = [
  {
    slug: "scientific-figure-pipeline",
    title: "Scientific figure pipeline",
    summary:
      "A workflow for turning papers, diagrams, and code outputs into publication-grade figures while preserving source traceability.",
    date: "2026-05-28",
    type: "project",
    collaborators: "AI tooling",
    image: "/assets/visuals/figure-field.png"
  },
  {
    slug: "notes-on-ai-agents",
    title: "Notes on agent reliability",
    summary:
      "Short observations on making agent workflows inspectable, resumable, and grounded in project-local conventions.",
    date: "2026-04-12",
    type: "note",
    collaborators: "Research memo"
  },
  {
    slug: "knowledge-interface-principles",
    title: "Knowledge interface principles",
    summary:
      "A compact set of design principles for dense, repeat-use research interfaces that prioritize scanning, comparison, and revision.",
    date: "2026-02-07",
    type: "note",
    collaborators: "Interface design"
  },
  {
    slug: "molecular-visualization-stack",
    title: "Molecular visualization stack",
    summary:
      "A practical stack for producing protein structure imagery and annotated scientific visuals from local data and public repositories.",
    date: "2025-11-16",
    type: "project",
    collaborators: "Life-science AI"
  }
];

export const entries: Entry[] = [...blogEntries, ...standaloneEntries];
