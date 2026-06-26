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
  newsletter: {
    buttondownUsername: "ecnu_enzyme",
    siteUrl: "https://jxtang.tech"
  },
  assets: {
    logo: "/assets/brand/brand-mark.svg",
    avatar: "/assets/brand/jianxin-tang.png"
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
    "I work at the intersection of AI for science, computational biology, and protein design. I am currently a PhD student at East China Normal University, with prior master's training in computational chemistry at East China University of Science and Technology.",
    "I use this site as a compact research index for writing, paper notes, tool-building, and selected work around scientific AI systems."
  ],
  readings: [] as Reading[],
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
      role: "Doctoral research",
      place: "East China Normal University",
      period: "2023 - Present",
      description:
        "Working on computational biology, protein design, and protein engineering."
    },
    {
      role: "Master's research",
      place: "East China University of Science and Technology",
      period: "2020 - 2023",
      description:
        "Focused on computational chemistry and molecular modeling-oriented research."
    },
    {
      role: "Industry internships",
      place: "BASF / Sinovation Ventures",
      period: "2021 - 2023",
      description:
        "Completed applied research and data-oriented internships."
    }
  ]
};

export type Profile = typeof profile;
