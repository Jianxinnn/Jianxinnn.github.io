import type { BlogSourceStatus } from "@/content/blog/types";

export const DEFAULT_BLOG_IMAGE = "/assets/visuals/profile-field.png";

export const blogConfig = {
  pageSize: 8,
  featuredLimit: 2,
  searchResultLimit: 8
} as const;

export const blogSourceStatuses = [
  {
    status: "original",
    label: "Original",
    title: "Original writing",
    description: "Notes, logs, and essays written directly for this site."
  },
  {
    status: "translation",
    label: "Translation",
    title: "Translations",
    description: "Chinese translations and bilingual adaptations with source attribution."
  },
  {
    status: "repost",
    label: "Repost",
    title: "Reposts",
    description: "Republished or mirrored pieces kept here for reading and indexing."
  },
  {
    status: "imported",
    label: "Imported",
    title: "Imported articles",
    description: "Standalone HTML or converted articles imported into the profile archive."
  }
] as const satisfies Array<{
  status: BlogSourceStatus;
  label: string;
  title: string;
  description: string;
}>;
