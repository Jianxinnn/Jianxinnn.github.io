import protectedPostRecords from "@/content/protected-posts.json";
import type { BlogLanguage, BlogPost } from "@/content/blog/types";

export type ProtectedPostRecord = {
  slug: string;
  title: string;
  summary: string;
  date: string;
  readingTime?: string;
  encryptedPath?: string;
  image?: string;
  badge?: string;
  category?: string;
  language?: BlogLanguage;
  tags?: string[];
  featured?: boolean;
  listed?: boolean;
  updated?: string;
};

export type ProtectedPost = BlogPost & {
  encryptedPath: string;
  sourceType: "encrypted";
};

const records = protectedPostRecords as ProtectedPostRecord[];

export const protectedPosts: ProtectedPost[] = records.map((post) => ({
  slug: post.slug,
  title: post.title,
  summary: post.summary,
  date: post.date,
  readingTime: post.readingTime ?? "Protected",
  sourceType: "encrypted",
  href: `/protected/${post.slug}/`,
  encryptedPath: post.encryptedPath ?? `/protected/${post.slug}.json`,
  badge: post.badge ?? "Protected",
  ...(post.image ? { image: post.image } : {}),
  ...(post.category ? { category: post.category } : {}),
  ...(post.language ? { language: post.language } : {}),
  ...(post.tags?.length ? { tags: post.tags } : {}),
  ...(post.featured ? { featured: true } : {}),
  ...(post.listed === false ? { listed: false } : {}),
  ...(post.updated ? { updated: post.updated } : {})
}));

export function getProtectedPost(slug: string) {
  return protectedPosts.find((post) => post.slug === slug);
}
