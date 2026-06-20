export type Paper = {
  year: string;
  type: string;
  venue: string;
  title: string;
  description: string;
  tags: string[];
  links: Array<{
    label: string;
    href: string;
  }>;
};

export const profile = {
  name: "Jianxin Tang",
  siteTitle: "Jianxin Tang",
  headerTagline: "AI4S · Research Tools",
  shortRole: "AI4S / Research Systems",
  role: "AI for science, agent workflows, and research software",
  location: "Shanghai",
  intro:
    "I build research-facing AI systems for scientific reading, biological data workflows, molecular visualization, and structured knowledge production.",
  about:
    "I work on practical AI-for-science systems and research tooling that help turn dense scientific material into usable software, figures, and decisions.",
  email: "jstangjianxin@163.com",
  assets: {
    logo: "/assets/brand/brand-mark.svg",
    avatar: "/assets/brand/jianxin-tang.jpeg"
  },
  nav: [
    { label: "Home", href: "/" },
    { label: "Blog", href: "/blog" },
    { label: "Notes", href: "/notes" },
    { label: "Papers", href: "/papers" },
    { label: "About", href: "/about" }
  ],
  links: [
    { label: "GitHub", href: "https://github.com/Jianxinnn" },
    {
      label: "Google Scholar",
      href: "https://scholar.google.com/citations?user=UYHlxe0AAAAJ&hl=zh-CN"
    }
  ],
  bio: [
    "I am interested in the tooling layer around scientific AI: literature workflows, biological data interfaces, molecular visualization, and agent systems that can operate inside real research projects.",
    "This site is organized as a personal research index rather than a newsletter. It keeps projects, notes, papers, and profile context separate so each section can grow without forcing everything into one feed."
  ],
  papers: [
    {
      year: "2025",
      type: "Paper note",
      venue: "AAAI 2025",
      title: "Controllable Protein Sequence Generation with LLM Preference Optimization",
      description:
        "Reading and implementation notes around controllable generation, preference signals, and protein sequence design.",
      tags: ["protein design", "preference optimization"],
      links: []
    },
    {
      year: "2025",
      type: "Paper note",
      venue: "arXiv",
      title: "A Variational Perspective on Generative Protein Fitness Optimization",
      description:
        "Notes on fitness optimization, generative modeling, and how optimization objectives shape sequence search.",
      tags: ["generative modeling", "fitness optimization"],
      links: []
    },
    {
      year: "2025",
      type: "Paper note",
      venue: "arXiv",
      title: "Genome modeling and design across all domains of life with Evo 2",
      description:
        "A running note on large biological sequence models and their implications for design workflows.",
      tags: ["genome models", "biological sequences"],
      links: []
    }
  ] satisfies Paper[],
  experience: [
    {
      role: "AI-for-science tooling",
      place: "Independent research and engineering",
      period: "2025 - Present",
      description:
        "Building systems for paper reading, figure generation, molecular visualization, biological data workflows, and agent-assisted research operations."
    },
    {
      role: "Agent workflow design",
      place: "Personal systems stack",
      period: "2024 - Present",
      description:
        "Designing reusable workflows for coding agents, research assistants, and structured project automation."
    },
    {
      role: "Scientific information design",
      place: "Research communication",
      period: "Ongoing",
      description:
        "Turning dense technical material into readable interfaces, diagrams, slides, and publication-style artifacts."
    }
  ]
};

export type Profile = typeof profile;
