export const blogCategories = [
  {
    slug: "technical-explainers",
    label: "Technical explainers",
    description: "Long-form translations, algorithm walkthroughs, and illustrated technical articles."
  },
  {
    slug: "research-systems",
    label: "Research systems",
    description: "Tools, workflows, and AI-for-science systems."
  },
  {
    slug: "build-logs",
    label: "Build logs",
    description: "Implementation notes and project updates."
  },
  {
    slug: "site-notes",
    label: "Site notes",
    description: "Design decisions and content-system notes for this profile."
  }
] as const;

export const allowedBlogCategories = blogCategories.map((category) => category.label);
