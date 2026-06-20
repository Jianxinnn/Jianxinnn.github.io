import { blogPosts } from "@/.generated/blog-posts";
import type { BlogPost } from "@/content/blog/types";

export type { BlogPost, BlogPostMeta, BlogSourceType } from "@/content/blog/types";
export { blogPosts };

export const BLOG_PAGE_SIZE = 2;

export function sortBlogPosts(posts: BlogPost[] = blogPosts) {
  return [...posts].sort(
    (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()
  );
}

export function getBlogPost(slug: string) {
  return blogPosts.find((post) => post.slug === slug);
}

export function getBlogPage(page: number, pageSize = BLOG_PAGE_SIZE) {
  const posts = sortBlogPosts(blogPosts);
  const totalPages = Math.max(1, Math.ceil(posts.length / pageSize));
  const currentPage = Math.min(Math.max(page, 1), totalPages);
  const start = (currentPage - 1) * pageSize;

  return {
    currentPage,
    posts: posts.slice(start, start + pageSize),
    totalPages
  };
}

export function slugifyTag(tag: string) {
  return tag
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

export function getBlogTags(posts: BlogPost[] = blogPosts) {
  const tags = new Map<string, { label: string; slug: string; count: number }>();

  for (const post of posts) {
    for (const tag of post.tags ?? []) {
      const slug = slugifyTag(tag);
      const current = tags.get(slug);

      tags.set(slug, {
        label: current?.label ?? tag,
        slug,
        count: (current?.count ?? 0) + 1
      });
    }
  }

  return [...tags.values()].sort((a, b) => a.label.localeCompare(b.label));
}

export function getPostsByTagSlug(tagSlug: string) {
  return sortBlogPosts(
    blogPosts.filter((post) =>
      post.tags?.some((tag) => slugifyTag(tag) === tagSlug)
    )
  );
}
