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
  featured?: boolean;
  pinned?: boolean;
};

export const entries: Entry[] = [
  {
    slug: "agentic-research-workbench",
    title: "Agentic Research Workbench",
    summary:
      "A modular workspace for reading papers, extracting structure, generating figures, and turning scattered research artifacts into reusable knowledge.",
    date: "2026-06-20",
    type: "project",
    collaborators: "Personal systems research",
    image: "/assets/visuals/profile-field.png",
    featured: true,
    pinned: true
  },
  {
    slug: "publication-profile-design",
    title: "A publication-first personal profile",
    summary:
      "Design notes on replacing the usual portfolio grid with a compact research publication architecture: featured work, dated archive, and long-form About.",
    date: "2026-06-18",
    type: "note",
    collaborators: "Design note",
    image: "/assets/visuals/notes-field.png",
    pinned: true
  },
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
    type: "writing",
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
