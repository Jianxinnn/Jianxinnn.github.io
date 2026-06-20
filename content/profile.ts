export const profile = {
  name: "Jianxin Tang",
  siteTitle: "Jianxin Tang",
  headerTagline: "AI for Science · Agent Systems",
  shortRole: "AI4S / Research Tooling",
  role: "AI systems, scientific tooling, and agentic workflows",
  location: "Shanghai",
  intro:
    "I build research-facing AI systems: tools for scientific reading, molecular visualization, agent workflows, and structured knowledge production.",
  about:
    "I work on practical AI-for-science systems, research tooling, and agentic workflows that help turn dense scientific material into usable software, figures, and decisions.",
  email: "jstangjianxin@163.com",
  assets: {
    logo: "/assets/brand/brand-mark.svg",
    avatar: "/assets/brand/avatar.svg"
  },
  nav: [
    { label: "Home", href: "/" },
    { label: "Notes", href: "/notes" },
    { label: "Archive", href: "/archive" },
    { label: "About", href: "/about" }
  ],
  links: [
    { label: "GitHub", href: "https://github.com/Jianxinnn" },
    { label: "Google Scholar", href: "https://scholar.google.com/" },
    { label: "LinkedIn", href: "https://www.linkedin.com/" }
  ],
  facts: [
    { label: "Focus", value: "AI for Science, agent systems, research software" },
    { label: "Base", value: "Shanghai" },
    { label: "Contact", value: "jstangjianxin@163.com" }
  ],
  bio: [
    "I am interested in the tooling layer around scientific AI: literature workflows, biological data interfaces, molecular visualization, and agent systems that can operate inside real research projects.",
    "This site is structured as a compact personal publication. Home highlights current work, Notes keeps short-form thinking, Archive records project history, and About gives the human context."
  ],
  papers: [
    {
      title: "Controllable Protein Sequence Generation with LLM Preference Optimization",
      venue: "Paper note / AAAI 2025",
      description:
        "Reading and implementation notes around controllable generation, preference signals, and protein sequence design."
    },
    {
      title: "A Variational Perspective on Generative Protein Fitness Optimization",
      venue: "Paper note / arXiv",
      description:
        "Notes on fitness optimization, generative modeling, and how optimization objectives shape sequence search."
    },
    {
      title: "Genome modeling and design across all domains of life with Evo 2",
      venue: "Paper note / arXiv",
      description:
        "A running note on large biological sequence models and their implications for design workflows."
    }
  ],
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
