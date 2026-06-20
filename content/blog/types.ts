export type BlogSourceType = "mdx" | "html" | "external";

export type BlogPostMeta = {
  title: string;
  summary: string;
  date: string;
  sourceType: BlogSourceType;
  href?: string;
  readingTime?: string;
  image?: string;
  badge?: string;
  tags?: string[];
  featured?: boolean;
};

export type BlogPost = Required<
  Pick<BlogPostMeta, "title" | "summary" | "date" | "sourceType" | "href" | "readingTime">
> &
  Pick<BlogPostMeta, "image" | "badge" | "tags" | "featured"> & {
    slug: string;
  };
