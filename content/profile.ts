export const profile = {
  name: "Jiexiang Tang",
  siteTitle: "Jiexiang Tang",
  role: "AI systems, scientific tooling, and agentic workflows",
  location: "Shanghai",
  intro:
    "I build research-facing AI systems: tools for scientific reading, molecular visualization, agent workflows, and structured knowledge production.",
  about:
    "This profile is structured as a living publication rather than a static resume. The homepage highlights current work, Notes captures short thinking, Archive keeps a dated record, and About provides the longer context.",
  email: "hello@example.com",
  nav: [
    { label: "Home", href: "/" },
    { label: "Notes", href: "/notes" },
    { label: "Archive", href: "/archive" },
    { label: "About", href: "/about" }
  ],
  links: [
    { label: "GitHub", href: "https://github.com/" },
    { label: "Google Scholar", href: "https://scholar.google.com/" },
    { label: "LinkedIn", href: "https://www.linkedin.com/" }
  ],
  aboutSections: [
    {
      title: "What I work on",
      body:
        "My work sits at the intersection of AI agents, scientific software, and high-density knowledge interfaces. I care about systems that help researchers move from raw material to usable insight with fewer manual seams."
    },
    {
      title: "Current focus",
      body:
        "I am currently shaping a personal stack for literature analysis, figure generation, biological data workflows, and AI-assisted product development."
    },
    {
      title: "Collaboration",
      body:
        "I am most interested in collaborations around research tooling, life-science AI, information design, and practical agent systems that can survive real workflows."
    }
  ]
};

export type Profile = typeof profile;
