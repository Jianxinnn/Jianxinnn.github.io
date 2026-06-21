import { blogPosts } from "@/content/blog/posts";
import { profile } from "@/content/profile";

export type EntryType = "project" | "note" | "writing" | "reading" | "talk" | "publication";

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
  updated?: string;
};

function slugify(value: string) {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

const blogEntries: Entry[] = blogPosts
  .filter((post) => post.listed !== false)
  .map((post) => ({
    slug: post.slug,
    title: post.title,
    summary: post.summary,
    date: post.date,
    type: "writing",
    collaborators: post.tags?.join(" / ") ?? post.readingTime,
    image: post.image,
    href: post.href,
    pinned: post.featured,
    updated: post.updated
  }));

const readingEntries: Entry[] = profile.readings.map((reading) => ({
  slug: `reading-${slugify(reading.title)}`,
  title: reading.title,
  summary: reading.description,
  date: `${reading.year}-01-01`,
  type: "reading",
  collaborators: reading.tags.join(" / "),
  href: "/readings"
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
    slug: "adaworld-latent-actions",
    title: "AdaWorld latent actions",
    summary:
      "A short note on action-aware pretraining as a reusable control interface for adaptable world models.",
    date: "2026-06-21",
    type: "note",
    collaborators: "Research memo",
    image: "/assets/visuals/profile-field.png",
    href: "/notes/adaworld-latent-actions/",
    updated: "2026-06-21T23:59:00+08:00"
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

export const entries: Entry[] = [...blogEntries, ...readingEntries, ...standaloneEntries];
