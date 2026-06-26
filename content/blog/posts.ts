import { blogPosts as generatedBlogPosts } from "@/.generated/blog-posts";
import { blogConfig, blogSourceStatuses } from "@/content/blog/config";
import { blogCategories } from "@/content/blog/categories";
import { protectedPosts } from "@/content/protected-posts";
import type { BlogPost, BlogSourceStatus } from "@/content/blog/types";

export type {
  BlogLanguage,
  BlogPost,
  BlogPostMeta,
  BlogPostSource,
  BlogSourceStatus,
  BlogSourceType
} from "@/content/blog/types";

export const blogPosts: BlogPost[] = [...generatedBlogPosts, ...protectedPosts];

export const BLOG_PAGE_SIZE = blogConfig.pageSize;
export const listedBlogPosts = blogPosts.filter((post) => post.listed !== false);

export function sortBlogPosts(posts: BlogPost[] = listedBlogPosts) {
  return [...posts].sort(
    (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()
  );
}

export function getBlogPost(slug: string) {
  return blogPosts.find((post) => post.slug === slug);
}

export function getBlogPage(page: number, pageSize = BLOG_PAGE_SIZE) {
  const posts = sortBlogPosts();
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

export function getBlogTags(posts: BlogPost[] = listedBlogPosts) {
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
    listedBlogPosts.filter((post) =>
      post.tags?.some((tag) => slugifyTag(tag) === tagSlug)
    )
  );
}

export function getFeaturedPosts(posts: BlogPost[] = listedBlogPosts) {
  return sortBlogPosts(posts.filter((post) => post.featured));
}

export function getBlogCategories(posts: BlogPost[] = listedBlogPosts) {
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

  return sortBlogPosts(listedBlogPosts.filter((post) => post.category === category.label));
}

export function getBlogSourceStatus(status: string) {
  return blogSourceStatuses.find((item) => item.status === status);
}

export function getPostsBySourceStatus(status: BlogSourceStatus) {
  return sortBlogPosts(
    listedBlogPosts.filter((post) => (post.source?.status ?? "original") === status)
  );
}

export function getBlogArchive(posts: BlogPost[] = listedBlogPosts) {
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

export function getBlogArchiveByMonth(posts: BlogPost[] = listedBlogPosts) {
  return sortBlogPosts(posts).reduce<
    Array<{
      year: string;
      months: Array<{ key: string; label: string; posts: BlogPost[] }>;
      posts: BlogPost[];
    }>
  >((groups, post) => {
    const date = new Date(post.date);
    const year = date.getFullYear().toString();
    const month = String(date.getMonth() + 1).padStart(2, "0");
    const monthKey = `${year}-${month}`;
    const yearGroup = groups.find((item) => item.year === year);
    const monthLabel = new Intl.DateTimeFormat("en", {
      month: "long"
    }).format(date);

    if (yearGroup) {
      yearGroup.posts.push(post);
      const monthGroup = yearGroup.months.find((item) => item.key === monthKey);

      if (monthGroup) {
        monthGroup.posts.push(post);
      } else {
        yearGroup.months.push({ key: monthKey, label: monthLabel, posts: [post] });
      }
    } else {
      groups.push({
        year,
        months: [{ key: monthKey, label: monthLabel, posts: [post] }],
        posts: [post]
      });
    }

    return groups;
  }, []);
}

export function getBlogSourceStats(posts: BlogPost[] = listedBlogPosts) {
  const counts = new Map<string, number>();

  for (const post of posts) {
    const status = post.source?.status ?? "original";
    counts.set(status, (counts.get(status) ?? 0) + 1);
  }

  return blogSourceStatuses
    .map((source) => ({
      ...source,
      count: counts.get(source.status) ?? 0
    }))
    .filter((source) => source.count > 0);
}
