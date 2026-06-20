import { blogPosts } from "@/.generated/blog-posts";
import { blogCategories } from "@/content/blog/categories";
import type { BlogPost } from "@/content/blog/types";

export type {
  BlogLanguage,
  BlogPost,
  BlogPostMeta,
  BlogPostSource,
  BlogSourceStatus,
  BlogSourceType
} from "@/content/blog/types";
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

export function slugifyCategory(category: string) {
  const configured = blogCategories.find((item) => item.label === category);

  return configured?.slug ?? slugifyTag(category);
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

export function getFeaturedPosts(posts: BlogPost[] = blogPosts) {
  return sortBlogPosts(posts.filter((post) => post.featured));
}

export function getBlogCategories(posts: BlogPost[] = blogPosts) {
  const counts = new Map<string, number>();

  for (const post of posts) {
    if (post.category) {
      counts.set(post.category, (counts.get(post.category) ?? 0) + 1);
    }
  }

  return blogCategories
    .map((category) => ({
      ...category,
      count: counts.get(category.label) ?? 0
    }))
    .filter((category) => category.count > 0);
}

export function getPostsByCategorySlug(categorySlug: string) {
  const category = blogCategories.find((item) => item.slug === categorySlug);

  if (!category) {
    return [];
  }

  return sortBlogPosts(blogPosts.filter((post) => post.category === category.label));
}

export function getBlogArchive(posts: BlogPost[] = blogPosts) {
  return sortBlogPosts(posts).reduce<Array<{ year: string; posts: BlogPost[] }>>(
    (groups, post) => {
      const year = new Date(post.date).getFullYear().toString();
      const group = groups.find((item) => item.year === year);

      if (group) {
        group.posts.push(post);
      } else {
        groups.push({ year, posts: [post] });
      }

      return groups;
    },
    []
  );
}

export function getBlogSourceStats(posts: BlogPost[] = blogPosts) {
  const labels: Record<string, string> = {
    imported: "Imported",
    original: "Original",
    repost: "Repost",
    translation: "Translation"
  };
  const counts = new Map<string, number>();

  for (const post of posts) {
    const status = post.source?.status ?? "original";
    counts.set(status, (counts.get(status) ?? 0) + 1);
  }

  return [...counts.entries()]
    .map(([status, count]) => ({
      count,
      label: labels[status] ?? status,
      status
    }))
    .sort((a, b) => a.label.localeCompare(b.label));
}
