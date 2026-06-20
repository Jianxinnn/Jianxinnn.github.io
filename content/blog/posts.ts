export type BlogSourceType = "mdx" | "html" | "external";

export type BlogPost = {
  slug: string;
  title: string;
  summary: string;
  date: string;
  readingTime: string;
  sourceType: BlogSourceType;
  href: string;
  image?: string;
  badge?: string;
  tags?: string[];
  featured?: boolean;
};

export const blogPosts: BlogPost[] = [
  {
    slug: "alphafold3-illustrated-cn",
    title: "图解 AlphaFold",
    summary:
      "一篇 AlphaFold3 架构的中文可视化导览，梳理输入准备、表征学习、结构预测与置信度评估等模块。",
    date: "2026-06-20",
    readingTime: "78 min read",
    sourceType: "html",
    href: "/blog/alphafold3-illustrated-cn/",
    image: "/assets/visuals/notes-field.png",
    badge: "转载 / 译",
    tags: ["AlphaFold3", "illustrated note"],
    featured: true
  },
  {
    slug: "agentic-research-workbench",
    title: "Agentic Research Workbench",
    summary:
      "A modular workspace for reading papers, extracting structure, generating figures, and turning scattered research artifacts into reusable knowledge.",
    date: "2026-06-20",
    readingTime: "4 min read",
    sourceType: "mdx",
    href: "/blog/agentic-research-workbench/",
    image: "/assets/visuals/profile-field.png",
    tags: ["research systems", "agents"],
    featured: true
  },
  {
    slug: "publication-profile-design",
    title: "A publication-first personal profile",
    summary:
      "Design notes on turning a personal site into a compact research index with separate spaces for papers, notes, projects, and profile context.",
    date: "2026-06-18",
    readingTime: "3 min read",
    sourceType: "mdx",
    href: "/blog/publication-profile-design/",
    image: "/assets/visuals/notes-field.png",
    tags: ["site design", "content system"]
  }
];

export function sortBlogPosts(posts = blogPosts) {
  return [...posts].sort(
    (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()
  );
}

export function getBlogPost(slug: string) {
  return blogPosts.find((post) => post.slug === slug);
}
