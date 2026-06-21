export type Reading = {
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

export type Publication = {
  authors: string;
  links: Array<{
    label: string;
    href: string;
  }>;
  title: string;
  venue: string;
  year: string;
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
  email: "jstangjianxin@163.com",
  assets: {
    logo: "/assets/brand/brand-mark.svg",
    avatar: "/assets/brand/jianxin-tang.jpeg"
  },
  nav: [
    { label: "Home", href: "/" },
    { label: "Blog", href: "/blog" },
    { label: "Notes", href: "/notes" },
    { label: "Readings", href: "/readings" },
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
    "This site is organized as a personal research index rather than a newsletter. It keeps long-form writing, short notes, paper readings, and profile context separate so each section can grow without forcing everything into one feed."
  ],
  readings: [
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
  ] satisfies Reading[],
  publications: [
    {
      year: "2026",
      title:
        "Artificial intelligence in oncology drug discovery: from target identification to therapeutic molecule generation",
      authors: "J Tang, J Xu, W Zhang, D Gong, Q Huang, X Cheng, H Li",
      venue: "Advanced Cancer Research 1 (2), 1-42",
      links: [
        {
          label: "Article",
          href: "https://doi.org/10.55092/acr20260005"
        }
      ]
    },
    {
      year: "2026",
      title:
        "Transformer-based multidimensional feature fusion for accurate prediction of lipid nanoparticles transfection efficiency",
      authors: "D Gong, X Xie, J Tang, S Li, H Li",
      venue: "Briefings in Bioinformatics 27 (2), bbag092",
      links: [
        {
          label: "Article",
          href: "https://doi.org/10.1093/bib/bbag092"
        }
      ]
    },
    {
      year: "2026",
      title:
        "Artificial intelligence in biologic drug discovery: A review of methodological evolution and therapeutic applications",
      authors: "J Tang, D Gong, H Li, S Li",
      venue: "Acta Pharmaceutica Sinica B",
      links: [
        {
          label: "Article",
          href: "https://doi.org/10.1016/j.apsb.2026.01.039"
        }
      ]
    },
    {
      year: "2025",
      title: "Method development for potential drug target identification and drug discovery",
      authors: "D Gong, J Tang, B Wang, S Xiang, Z Feng, S Li, H Li",
      venue: "Scientia Sinica Chimica 55 (8), 2223-2242",
      links: [
        {
          label: "Article",
          href: "https://doi.org/10.1360/SSC-2025-0118"
        }
      ]
    },
    {
      year: "2024",
      title:
        "EvoLlama: Enhancing LLMs' Understanding of Proteins via Multimodal Structure and Sequence Representations",
      authors: "N Liu, C Sun, T Ji, J Tian, J Tang, Y Wu, M Lan",
      venue: "arXiv preprint arXiv:2412.11618",
      links: [
        {
          label: "arXiv",
          href: "https://arxiv.org/abs/2412.11618"
        }
      ]
    }
  ] satisfies Publication[],
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
